"""Quest 3 input device for teleoperation via Hand Tracking Streamer (HTS).

Receives hand tracking data from Meta Quest 3 over UDP or TCP.
HTS sends UTF-8 CSV lines with 21 hand landmarks and wrist pose.

Usage:
    python teleop_sim.py --input quest3 --port 9000
    python teleop_sim.py --input quest3 --port 8000 --protocol tcp
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from anydexretarget.mediapipe import MediaPipeSmoother

logger = logging.getLogger(__name__)

# Unity LH (x right, y up, z forward) -> RH (x front, y left, z up)
_UNITY_TO_RH = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    """Return a normalized quaternion."""
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return quat / norm


def _quat_rotate(points: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Rotate Nx3 points by quaternion (x, y, z, w)."""
    quat = _quat_normalize(quat)
    q_xyz = quat[:3]
    q_w = quat[3]
    t = 2.0 * np.cross(q_xyz, points)
    return points + q_w * t + np.cross(q_xyz, t)


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    quat = _quat_normalize(quat)
    x, y, z, w = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=float,
    )


def _matrix_to_quat(mat: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a quaternion (x, y, z, w)."""
    trace = mat[0, 0] + mat[1, 1] + mat[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (mat[2, 1] - mat[1, 2]) * s
        y = (mat[0, 2] - mat[2, 0]) * s
        z = (mat[1, 0] - mat[0, 1]) * s
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
        w = (mat[2, 1] - mat[1, 2]) / s
        x = 0.25 * s
        y = (mat[0, 1] + mat[1, 0]) / s
        z = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
        w = (mat[0, 2] - mat[2, 0]) / s
        x = (mat[0, 1] + mat[1, 0]) / s
        y = 0.25 * s
        z = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
        w = (mat[1, 0] - mat[0, 1]) / s
        x = (mat[0, 2] + mat[2, 0]) / s
        y = (mat[1, 2] + mat[2, 1]) / s
        z = 0.25 * s
    return _quat_normalize(np.array([x, y, z, w], dtype=float))


def _convert_vec(vec: np.ndarray) -> np.ndarray:
    """Convert a vector from Unity LH to RH coordinates."""
    return _UNITY_TO_RH @ vec


def _convert_quat(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion from Unity LH to RH coordinates."""
    r_unity = _quat_to_matrix(quat)
    r_rh = _UNITY_TO_RH @ r_unity @ _UNITY_TO_RH.T
    return _matrix_to_quat(r_rh)


@dataclass
class _HandState:
    """State for a single hand."""

    side: str
    wrist_position: Optional[np.ndarray] = None
    wrist_quat: Optional[np.ndarray] = None
    landmarks_local: Optional[np.ndarray] = None
    last_update: float = field(default_factory=time.monotonic)

    def update_wrist(self, data: Iterable[float]) -> None:
        values = np.array(list(data), dtype=float)
        if values.size < 7:
            return
        self.wrist_position = _convert_vec(values[:3])
        self.wrist_quat = _convert_quat(values[3:7])
        self.last_update = time.monotonic()

    def update_landmarks(self, data: Iterable[float]) -> None:
        values = np.array(list(data), dtype=float)
        if values.size < 3:
            return
        if values.size % 3 != 0:
            values = values[: values.size - (values.size % 3)]
        reshaped = values.reshape((-1, 3))
        self.landmarks_local = (_UNITY_TO_RH @ reshaped.T).T
        self.last_update = time.monotonic()

    def world_points(self) -> Optional[np.ndarray]:
        """Return landmarks transformed to world space (N, 3)."""
        if self.landmarks_local is None:
            return None
        if self.wrist_position is None or self.wrist_quat is None:
            return self.landmarks_local
        return _quat_rotate(self.landmarks_local, self.wrist_quat) + self.wrist_position


def _parse_line(line: str) -> Optional[Tuple[str, str, Tuple[float, ...]]]:
    """Parse a CSV line into (side, kind, floats)."""
    parts = [part.strip() for part in line.split(",")]
    if not parts:
        return None
    label = parts[0].lower()
    if "wrist" not in label and "landmarks" not in label:
        return None
    side = "right" if "right" in label else "left" if "left" in label else ""
    if not side:
        return None
    kind = "wrist" if "wrist" in label else "landmarks"
    floats = []
    for part in parts[1:]:
        if not part:
            continue
        try:
            floats.append(float(part))
        except ValueError:
            continue
    return (side, kind, tuple(floats))


class Quest3:
    """Quest 3 hand tracking input device via HTS UDP/TCP stream.

    Receives hand landmark data from the Hand Tracking Streamer app running
    on Meta Quest 3. The 21 HTS joints map directly to MediaPipe 21 keypoints.

    Args:
        host: Host/IP to bind the listener to.
        port: Port to listen on.
        protocol: Transport protocol, "udp" or "tcp".
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9000,
        protocol: str = "udp",
    ) -> None:
        self._host = host
        self._port = port
        self._protocol = protocol.lower()
        if self._protocol not in ("udp", "tcp"):
            raise ValueError(f"protocol must be 'udp' or 'tcp', got '{protocol}'")

        self._hands: Dict[str, _HandState] = {
            "right": _HandState("right"),
            "left": _HandState("left"),
        }
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._smoother = MediaPipeSmoother(buffer_size=5)

        # Start receiver thread
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._conn_threads: list[threading.Thread] = []
        self._thread.start()
        logger.info(
            "Quest3 receiver started (%s %s:%d)", self._protocol, self._host, self._port
        )

    # -- public API (duck-typed with VisionPro / MediaPipeReplay) --

    def get_fingers_data(self) -> dict:
        """Return finger data in MediaPipe (21, 3) format.

        Returns:
            dict with "left_fingers" and "right_fingers" np.ndarray (21, 3).
        """
        empty = np.zeros((21, 3), dtype=np.float32)
        with self._lock:
            left_pts = self._hands["left"].world_points()
            right_pts = self._hands["right"].world_points()

        left = left_pts.astype(np.float32) if left_pts is not None else empty.copy()
        right = right_pts.astype(np.float32) if right_pts is not None else empty.copy()

        return {
            "left_fingers": left,
            "right_fingers": right,
        }

    def stop(self) -> None:
        """Stop the receiver thread."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        for t in list(self._conn_threads):
            if t.is_alive():
                t.join(timeout=0.5)

    # -- internal receiver --

    def _handle_line(self, line: str) -> None:
        parsed = _parse_line(line)
        if not parsed:
            return
        side, kind, floats = parsed
        with self._lock:
            hand = self._hands[side]
            if kind == "wrist":
                hand.update_wrist(floats)
            elif kind == "landmarks":
                hand.update_landmarks(floats)

    def _run(self) -> None:
        if self._protocol == "udp":
            self._run_udp()
        else:
            self._run_tcp()

    def _run_udp(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, self._port))
        sock.settimeout(0.5)
        logger.info("UDP listening on %s:%d", self._host, self._port)
        try:
            while not self._stop.is_set():
                try:
                    data, _addr = sock.recvfrom(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                try:
                    message = data.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                for line in message.splitlines():
                    if line:
                        self._handle_line(line)
        finally:
            sock.close()

    def _run_tcp(self) -> None:
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self._host, self._port))
        server_sock.listen(1)
        server_sock.settimeout(0.5)
        logger.info("TCP server listening on %s:%d", self._host, self._port)
        try:
            while not self._stop.is_set():
                try:
                    conn, addr = server_sock.accept()
                except socket.timeout:
                    continue
                thread = threading.Thread(
                    target=self._handle_tcp_conn, args=(conn, addr), daemon=True
                )
                self._conn_threads.append(thread)
                thread.start()
        finally:
            server_sock.close()

    def _handle_tcp_conn(self, conn: socket.socket, addr) -> None:
        with conn:
            logger.info("Accepted TCP connection from %s", addr)
            conn.settimeout(0.5)
            buffer = ""
            while not self._stop.is_set():
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    continue
                if not data:
                    break
                try:
                    buffer += data.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line:
                        self._handle_line(line)

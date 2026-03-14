"""Teleoperate a real Kinova Gen3 arm + real Wuji Hand via Quest 3 hand tracking.

Design:
- Quest wrist pose -> residual target pose -> MuJoCo IK -> Kinova joint-speed commands
- Quest 21 landmarks -> retargeting -> Wuji Hand 20 joint targets
- MuJoCo is used only as a kinematic / IK model, not as a simulator

Example:
    python teleop_env/teleop_kinova_wuji_real.py --port 9000 --kinova-ip 192.168.1.10
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import math
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from wuji_retargeting import Retargeter

from util.ik import solve_pose_ik
from util.quaternion import (
    matrix_to_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_euler_xyz,
    transform_vr_to_robot_pose,
)
from util.udp_socket import (
    parse_left_landmarks,
    parse_left_wrist_pose,
    parse_right_landmarks,
    parse_right_wrist_pose,
)

try:
    import wujihandpy
except ImportError:  # pragma: no cover - depends on local hardware env
    wujihandpy = None

_KORTEX_EXAMPLES_DIR = (
    _Path(__file__).resolve().parents[1]
    / "Kinova-kortex2_Gen3_G3L"
    / "api_python"
    / "examples"
)
if str(_KORTEX_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_KORTEX_EXAMPLES_DIR))

import utilities as kortex_utilities  # noqa: E402
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient  # noqa: E402
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient  # noqa: E402
from kortex_api.autogen.messages import Base_pb2  # noqa: E402

# Gen3 home pose used only as an IK regularization prior.
_HOME_QPOS = np.array(
    [0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109, 1.57079633],
    dtype=np.float64,
)

_NUM_ARM_JOINTS = 7
_NUM_HAND_JOINTS = 20

# Unity LH (x right, y up, z forward) -> RH (x front, y left, z up)
_UNITY_TO_RH = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=float,
)


def _landmarks_to_mediapipe(raw_landmarks: list[float]) -> np.ndarray:
    arr = np.array(raw_landmarks, dtype=np.float64).reshape(21, 3)
    return (_UNITY_TO_RH @ arr.T).T


def _default_scene_path() -> Path:
    return Path(__file__).resolve().parent / "scene" / "scene_kinova_gen3_wuji.xml"


def _default_arm_scene_path() -> Path:
    return Path(__file__).resolve().parent / "scene" / "scene_kinova_gen3.xml"


def _default_hand_config_path() -> Path:
    return (
        _Path(__file__).resolve().parents[1]
        / ".."
        / "wuji-retargeting"
        / "example"
        / "config"
        / "adaptive_analytical_quest3.yaml"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teleop real Kinova Gen3 + Wuji Hand with Quest wrist residuals + retargeting."
    )
    parser.add_argument(
        "--scene",
        default=str(_default_scene_path()),
        help="Path to MuJoCo XML used only for kinematics / IK.",
    )
    parser.add_argument("--port", type=int, default=9000, help="UDP port to listen on.")
    parser.add_argument("--site", default="kinova_ee_site", help="End-effector site name.")
    parser.add_argument(
        "--hand-config",
        default=None,
        help="Path to retargeter YAML config. Default: auto-detect.",
    )
    parser.add_argument(
        "--hand-side",
        default="right",
        choices=["left", "right"],
        help="Hand side for both parsing and retargeting.",
    )
    parser.add_argument(
        "--position-scale",
        type=float,
        default=1.5,
        help="Scale for wrist position residuals.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.8,
        help="EMA smoothing factor for wrist residuals (0-1).",
    )
    parser.add_argument(
        "--wrist-pos-deadband",
        type=float,
        default=0.02,
        help="Ignore wrist position residuals whose norm is below this threshold in meters.",
    )
    parser.add_argument(
        "--wrist-rot-deadband-deg",
        type=float,
        default=8.0,
        help="Ignore wrist rotation residuals smaller than this threshold in degrees.",
    )
    parser.add_argument(
        "--rot-weight",
        type=float,
        default=1.0,
        help="Weight for orientation error in IK.",
    )
    parser.add_argument(
        "--enable-wrist-rotation",
        action="store_true",
        help="Enable wrist orientation tracking. Disabled by default for safer bring-up.",
    )
    parser.add_argument(
        "--ik-damping",
        type=float,
        default=1e-3,
        help="Damping factor for IK solver.",
    )
    parser.add_argument(
        "--ik-current-weight",
        type=float,
        default=0.1,
        help="Weight for penalizing deviation from current pose in IK.",
    )
    parser.add_argument(
        "--kinova-ip",
        default="192.168.1.10",
        help="Kinova robot IP.",
    )
    parser.add_argument(
        "--kinova-username",
        default="admin",
        help="Kinova username.",
    )
    parser.add_argument(
        "--kinova-password",
        default="admin",
        help="Kinova password.",
    )
    parser.add_argument(
        "--arm-kp",
        type=float,
        default=1.5,
        help="P gain from joint-angle error (deg) to joint speed (deg/s).",
    )
    parser.add_argument(
        "--arm-max-speed-deg",
        type=float,
        default=15.0,
        help="Per-joint speed limit in deg/s.",
    )
    parser.add_argument(
        "--arm-deadband-deg",
        type=float,
        default=0.5,
        help="Command zero speed if joint error magnitude is below this threshold.",
    )
    parser.add_argument(
        "--control-period",
        type=float,
        default=0.02,
        help="Main loop period in seconds.",
    )
    parser.add_argument(
        "--packet-timeout",
        type=float,
        default=0.25,
        help="Stop the arm if no valid Quest packet arrives within this many seconds.",
    )
    parser.add_argument(
        "--hand-cutoff-freq",
        type=float,
        default=5.0,
        help="Wuji hand low-pass filter cutoff frequency.",
    )
    parser.add_argument(
        "--disable-arm",
        action="store_true",
        help="Do not send commands to Kinova arm.",
    )
    parser.add_argument(
        "--disable-hand",
        action="store_true",
        help="Do not send commands to Wuji hand.",
    )
    parser.add_argument(
        "--move-home-on-start",
        action="store_true",
        help="Move Kinova to its saved Home action before starting teleoperation.",
    )
    parser.add_argument(
        "--home-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for the startup Home action.",
    )
    return parser.parse_args()


def _angle_error_deg(target_deg: np.ndarray, current_deg: np.ndarray) -> np.ndarray:
    return (target_deg - current_deg + 180.0) % 360.0 - 180.0


def _recv_latest_packet(sock: socket.socket) -> bytes | None:
    latest = None
    while True:
        try:
            latest, _ = sock.recvfrom(65536)
        except BlockingIOError:
            break
    return latest


def _make_socket(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.setblocking(False)
    return sock


def _make_hand_controller(args: argparse.Namespace):
    if args.disable_hand:
        return None, None
    if wujihandpy is None:
        raise ImportError(
            "wujihandpy is not installed, but Wuji hand control is enabled. "
            "Install it or pass --disable-hand."
        )
    hand = wujihandpy.Hand()
    hand.write_joint_enabled(True)
    controller = hand.realtime_controller(
        enable_upstream=False,
        filter=wujihandpy.filter.LowPass(cutoff_freq=args.hand_cutoff_freq),
    )
    time.sleep(0.5)
    return hand, controller


def _load_retargeter(args: argparse.Namespace) -> Retargeter | None:
    if args.disable_hand:
        return None
    hand_config = args.hand_config
    if hand_config is None:
        default_cfg = _default_hand_config_path()
        if default_cfg.exists():
            hand_config = str(default_cfg)
        else:
            print(f"Warning: default hand config not found at {default_cfg}")
            print("Hand retargeting will be disabled. Use --hand-config to specify.")
            return None
    retargeter = Retargeter.from_yaml(str(hand_config), args.hand_side)
    print(f"Retargeter loaded from {hand_config}")
    return retargeter


def _select_parsers(hand_side: str):
    if hand_side == "left":
        return parse_left_landmarks, parse_left_wrist_pose
    return parse_right_landmarks, parse_right_wrist_pose


def _get_measured_q_rad(base_cyclic: BaseCyclicClient) -> np.ndarray:
    feedback = base_cyclic.RefreshFeedback()
    q_deg = np.array(
        [feedback.actuators[i].position for i in range(_NUM_ARM_JOINTS)],
        dtype=np.float64,
    )
    # Kinova often reports angles in [0, 360). Convert to signed degrees so they
    # match the MuJoCo joint convention expected by the kinematic model / IK.
    q_deg = np.where(q_deg > 180.0, q_deg - 360.0, q_deg)
    return np.deg2rad(q_deg)


def _build_joint_speeds_command(speed_deg_s: np.ndarray):
    joint_speeds = Base_pb2.JointSpeeds()
    for i, speed in enumerate(speed_deg_s.tolist()):
        joint_speed = joint_speeds.joint_speeds.add()
        joint_speed.joint_identifier = i
        joint_speed.value = float(speed)
        joint_speed.duration = 0
    return joint_speeds


def _stop_arm(base: BaseClient) -> None:
    try:
        base.Stop()
    except Exception as exc:  # pragma: no cover - hardware dependent
        print(f"Warning: failed to stop Kinova arm cleanly: {exc}")


def _apply_position_deadband(vec: np.ndarray, deadband: float) -> np.ndarray:
    if deadband <= 0.0:
        return vec
    if np.linalg.norm(vec) < deadband:
        return np.zeros_like(vec)
    return vec


def _quaternion_rotation_angle_rad(q: tuple[float, float, float, float]) -> float:
    x, y, z, w = q
    sin_half = math.sqrt(x * x + y * y + z * z)
    return 2.0 * math.atan2(sin_half, abs(w))


def _apply_rotation_deadband(
    q: tuple[float, float, float, float], deadband_deg: float
) -> tuple[float, float, float, float]:
    if deadband_deg <= 0.0:
        return q
    deadband_rad = math.radians(deadband_deg)
    if _quaternion_rotation_angle_rad(q) < deadband_rad:
        return (0.0, 0.0, 0.0, 1.0)
    return q


def _check_for_end_or_abort(event: threading.Event):
    def check(notification, event=event):
        if (
            notification.action_event == Base_pb2.ACTION_END
            or notification.action_event == Base_pb2.ACTION_ABORT
        ):
            event.set()

    return check


def _move_arm_home(base: BaseClient, timeout: float = 30.0) -> bool:
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)

    action_handle = None
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle
            break

    if action_handle is None:
        print("Warning: Kinova Home action not found; skipping move-to-home.")
        return False

    finished = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        _check_for_end_or_abort(finished),
        Base_pb2.NotificationOptions(),
    )
    try:
        print("Moving Kinova arm to Home...")
        base.ExecuteActionFromReference(action_handle)
        ok = finished.wait(timeout)
        if ok:
            print("Kinova Home reached.")
        else:
            print("Warning: timeout while waiting for Kinova Home action.")
        return ok
    finally:
        base.Unsubscribe(notification_handle)


def main() -> None:
    args = _parse_args()

    default_full_scene = _default_scene_path().resolve()
    if args.disable_hand and Path(args.scene).expanduser().resolve() == default_full_scene:
        args.scene = str(_default_arm_scene_path())
        print(f"Arm-only mode: using arm scene {args.scene}")

    xml_path = Path(args.scene).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    state_data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, args.site)
    if site_id == -1:
        raise ValueError(f"Site '{args.site}' not found in model.")
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    sock = _make_socket(args.port)
    retargeter = _load_retargeter(args)
    parse_landmarks, parse_wrist_pose = _select_parsers(args.hand_side)

    hand = None
    hand_controller = None

    kinova_args = SimpleNamespace(
        ip=args.kinova_ip,
        username=args.kinova_username,
        password=args.kinova_password,
    )

    if args.disable_arm:
        raise ValueError("This script currently requires Kinova arm feedback for IK. Remove --disable-arm.")

    with kortex_utilities.DeviceConnection.createTcpConnection(kinova_args) as router, \
         kortex_utilities.DeviceConnection.createUdpConnection(kinova_args) as router_rt:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router_rt)

        servo_mode = Base_pb2.ServoingModeInformation()
        servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(servo_mode)

        if args.move_home_on_start:
            _move_arm_home(base, timeout=args.home_timeout)

        current_q_rad = _get_measured_q_rad(base_cyclic)
        if current_q_rad.shape[0] != _NUM_ARM_JOINTS:
            raise RuntimeError("Expected 7 Kinova joint readings.")

        state_q = np.array(model.qpos0, dtype=np.float64)
        state_q[:_NUM_ARM_JOINTS] = current_q_rad
        state_data.qpos[: model.nq] = state_q
        state_data.qvel[:] = 0.0
        mujoco.mj_forward(model, state_data)

        initial_site_pos = state_data.site_xpos[site_id].copy()
        initial_site_quat = matrix_to_quaternion(
            state_data.site_xmat[site_id].reshape(3, 3).copy()
        )
        base_xmat = None
        if base_body_id != -1:
            base_xmat = state_data.xmat[base_body_id].reshape(3, 3).copy()

        hand, hand_controller = _make_hand_controller(args)

        initial_wrist_position = None
        initial_wrist_quaternion = None
        target_position = initial_site_pos.copy()
        target_quaternion = np.array(initial_site_quat, dtype=np.float64)
        latest_hand_qpos = None
        latest_residual = None
        latest_euler_residual = None
        smoothed_residual = None
        last_log_time = time.time()
        last_valid_packet_time = 0.0

        print(f"  Initial arm q (deg): {np.rad2deg(current_q_rad).tolist()}")
        print(f"  HOME_QPOS (deg):     {np.rad2deg(_HOME_QPOS).tolist()}")
        print(f"  Initial EE pos:      {initial_site_pos.tolist()}")
        print(f"  model.nq={model.nq}  model.nv={model.nv}")
        print("Starting real teleoperation loop...")
        print(f"  Kinova IP: {args.kinova_ip}")
        print(f"  Quest UDP port: {args.port}")
        print(f"  Hand side: {args.hand_side}")
        print(f"  Arm speed limit: ±{args.arm_max_speed_deg:.1f} deg/s")
        print(f"  Packet timeout: {args.packet_timeout:.3f} s")
        print(f"  Wrist position deadband: {args.wrist_pos_deadband:.3f} m")
        print(f"  Wrist rotation deadband: {args.wrist_rot_deadband_deg:.1f} deg")
        print(f"  Wrist rotation tracking: {'ON' if args.enable_wrist_rotation else 'OFF'}")
        print("Press Ctrl+C to stop.")

        try:
            while True:
                loop_start = time.time()
                packet = _recv_latest_packet(sock)

                if packet is not None:
                    message = packet.decode("utf-8", errors="ignore")
                    saw_valid_data = False

                    if retargeter is not None:
                        landmarks = parse_landmarks(message)
                        if landmarks is not None:
                            mediapipe_pts = _landmarks_to_mediapipe(landmarks)
                            if not np.allclose(mediapipe_pts, 0):
                                latest_hand_qpos = retargeter.retarget(mediapipe_pts)
                                saw_valid_data = True

                    wrist_pose = parse_wrist_pose(message)
                    if wrist_pose is not None:
                        wrist_position = (wrist_pose[0], wrist_pose[1], wrist_pose[2])
                        wrist_quaternion = (
                            wrist_pose[3],
                            wrist_pose[4],
                            wrist_pose[5],
                            wrist_pose[6],
                        )
                        robot_position, robot_quaternion = transform_vr_to_robot_pose(
                            wrist_position, wrist_quaternion
                        )
                        saw_valid_data = True
                        if initial_wrist_position is None:
                            initial_wrist_position = robot_position
                            initial_wrist_quaternion = robot_quaternion
                            print("Captured initial wrist reference pose.")
                        else:
                            residual = np.array(
                                [
                                    robot_position[0] - initial_wrist_position[0],
                                    robot_position[1] - initial_wrist_position[1],
                                    robot_position[2] - initial_wrist_position[2],
                                ],
                                dtype=np.float64,
                            )
                            if base_xmat is not None:
                                residual = base_xmat @ residual
                            if smoothed_residual is None:
                                smoothed_residual = residual
                            else:
                                smoothed_residual = (
                                    args.ema_alpha * residual
                                    + (1.0 - args.ema_alpha) * smoothed_residual
                                )
                            smoothed_residual = _apply_position_deadband(
                                smoothed_residual, args.wrist_pos_deadband
                            )
                            target_position = (
                                initial_site_pos + args.position_scale * smoothed_residual
                            )

                            relative_quaternion = quaternion_multiply(
                                robot_quaternion,
                                quaternion_inverse(initial_wrist_quaternion),
                            )
                            # Empirical correction matching the existing Gen3 teleop setup.
                            relative_quaternion = (
                                -relative_quaternion[0],
                                -relative_quaternion[1],
                                relative_quaternion[2],
                                relative_quaternion[3],
                            )
                            if args.enable_wrist_rotation:
                                relative_quaternion = _apply_rotation_deadband(
                                    relative_quaternion, args.wrist_rot_deadband_deg
                                )
                                target_quaternion = np.array(
                                    quaternion_multiply(relative_quaternion, initial_site_quat),
                                    dtype=np.float64,
                                )
                                norm = np.linalg.norm(target_quaternion)
                                if norm > 0.0:
                                    target_quaternion /= norm
                            else:
                                relative_quaternion = (0.0, 0.0, 0.0, 1.0)
                                target_quaternion = np.array(
                                    initial_site_quat, dtype=np.float64
                                )
                            latest_residual = smoothed_residual
                            latest_euler_residual = quaternion_to_euler_xyz(
                                relative_quaternion[0],
                                relative_quaternion[1],
                                relative_quaternion[2],
                                relative_quaternion[3],
                            )

                    if saw_valid_data:
                        last_valid_packet_time = loop_start

                now = time.time()
                if (
                    latest_residual is not None
                    and latest_euler_residual is not None
                    and now - last_log_time > 1.0
                ):
                    print(
                        f"Wrist residual (xyz): {latest_residual.tolist()} "
                        f"euler: {list(latest_euler_residual)}"
                    )
                    last_log_time = now

                if latest_hand_qpos is not None and hand_controller is not None:
                    hand_controller.set_joint_target_position(
                        np.asarray(latest_hand_qpos, dtype=np.float64).reshape(5, 4)
                    )

                current_q_rad = _get_measured_q_rad(base_cyclic)
                current_q_deg = np.rad2deg(current_q_rad)

                if (
                    initial_wrist_position is None
                    or now - last_valid_packet_time > args.packet_timeout
                ):
                    base.SendJointSpeedsCommand(
                        _build_joint_speeds_command(np.zeros(_NUM_ARM_JOINTS, dtype=np.float64))
                    )
                else:
                    # Start from model default qpos (has valid quaternions for
                    # free joints like the target_cube) and override arm joints.
                    q_init = np.array(model.qpos0, dtype=np.float64)
                    q_init[:_NUM_ARM_JOINTS] = current_q_rad
                    q_sol = solve_pose_ik(
                        model,
                        ik_data,
                        site_id,
                        target_position,
                        target_quaternion,
                        q_init,
                        rot_weight=args.rot_weight,
                        damping=args.ik_damping,
                        current_q_weight=args.ik_current_weight,
                        home_qpos=_HOME_QPOS,
                        skip_tail_joints=0,
                    )
                    q_target_deg = np.rad2deg(q_sol[:_NUM_ARM_JOINTS])
                    q_err_deg = _angle_error_deg(q_target_deg, current_q_deg)
                    speed_deg_s = args.arm_kp * q_err_deg
                    speed_deg_s[np.abs(q_err_deg) < args.arm_deadband_deg] = 0.0
                    speed_deg_s = np.clip(
                        speed_deg_s,
                        -args.arm_max_speed_deg,
                        args.arm_max_speed_deg,
                    )
                    if now - last_log_time < 0.1:  # print once near each log
                        print(
                            f"  IK target pos: {target_position.tolist()}\n"
                            f"  q_current(deg): {current_q_deg.tolist()}\n"
                            f"  q_target (deg): {q_target_deg.tolist()}\n"
                            f"  q_err    (deg): {q_err_deg.tolist()}\n"
                            f"  speed  (deg/s): {speed_deg_s.tolist()}"
                        )
                    base.SendJointSpeedsCommand(_build_joint_speeds_command(speed_deg_s))

                elapsed = time.time() - loop_start
                sleep_time = args.control_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopping teleoperation...")
        finally:
            _stop_arm(base)
            if hand is not None:
                try:
                    hand.write_joint_enabled(False)
                except Exception as exc:  # pragma: no cover - hardware dependent
                    print(f"Warning: failed to disable Wuji hand cleanly: {exc}")
            sock.close()


if __name__ == "__main__":
    main()

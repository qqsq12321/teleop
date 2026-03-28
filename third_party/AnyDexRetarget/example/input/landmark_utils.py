"""Shared constants and functions for MediaPipe hand landmark processing.

All example input devices and test scripts that process MediaPipe landmarks
should import from here to avoid code duplication.
"""

import cv2
import numpy as np

# Reference palm size: wrist to middle MCP distance (meters)
REFERENCE_WRIST_TO_MIDDLE_MCP = 0.092

# Reference finger segment lengths (meters) from AVP stereo tracking data.
REFERENCE_SEGMENT_LENGTHS = {
    'thumb':  [0.0505, 0.0318, 0.0302],
    'index':  [0.0418, 0.0243, 0.0223],
    'middle': [0.0489, 0.0289, 0.0227],
    'ring':   [0.0422, 0.0274, 0.0227],
    'pinky':  [0.0343, 0.0195, 0.0201],
}

# MediaPipe landmark index groups per finger: [MCP, PIP, DIP, TIP]
FINGER_INDICES = {
    'thumb':  [1, 2, 3, 4],
    'index':  [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring':   [13, 14, 15, 16],
    'pinky':  [17, 18, 19, 20],
}

# MediaPipe hand skeleton connections (pairs of landmark indices)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def landmarks_to_array(hand_landmarks, w: int, h: int, z_scale: float = 2.5) -> np.ndarray:
    """Convert MediaPipe hand landmarks to a (21, 3) pixel-space array.

    Args:
        hand_landmarks: MediaPipe hand landmarks object.
        w: Frame width in pixels.
        h: Frame height in pixels.
        z_scale: Multiplier for z coordinate (default 2.5).

    Returns:
        (21, 3) float32 array in pixel coordinates.
    """
    kp = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )
    kp[:, 0] *= w
    kp[:, 1] *= h
    kp[:, 2] *= w * z_scale
    return kp


def correct_segment_lengths(kp: np.ndarray) -> np.ndarray:
    """Correct finger segment lengths to reference AVP measurements.

    Adjusts each finger segment to match the reference lengths while
    preserving the segment directions.

    Args:
        kp: (21, 3) keypoints array.

    Returns:
        (21, 3) corrected keypoints.
    """
    kp_corrected = kp.copy()
    for finger_name, indices in FINGER_INDICES.items():
        ref_lengths = REFERENCE_SEGMENT_LENGTHS[finger_name]
        mcp_i, pip_i, dip_i, tip_i = indices
        base = kp_corrected[mcp_i].copy()

        seg1 = kp[pip_i] - kp[mcp_i]
        seg1_len = np.linalg.norm(seg1)
        if seg1_len > 1e-6:
            kp_corrected[pip_i] = base + (seg1 / seg1_len) * ref_lengths[0]

        seg2 = kp[dip_i] - kp[pip_i]
        seg2_len = np.linalg.norm(seg2)
        if seg2_len > 1e-6:
            kp_corrected[dip_i] = kp_corrected[pip_i] + (seg2 / seg2_len) * ref_lengths[1]

        seg3 = kp[tip_i] - kp[dip_i]
        seg3_len = np.linalg.norm(seg3)
        if seg3_len > 1e-6:
            kp_corrected[tip_i] = kp_corrected[dip_i] + (seg3 / seg3_len) * ref_lengths[2]

    return kp_corrected


def process_landmarks(
    kp: np.ndarray,
    depth_scale: float = 1.25,
    correct_segments: bool = True,
    reference_wrist_to_mid_mcp: float = REFERENCE_WRIST_TO_MIDDLE_MCP,
) -> np.ndarray:
    """Full landmark post-processing: normalize, correct segments, scale depth.

    Args:
        kp: (21, 3) raw keypoints in pixel space.
        depth_scale: Multiplier for z/depth axis (default 1.25).
        correct_segments: Whether to apply segment length correction.
        reference_wrist_to_mid_mcp: Reference wrist-to-middle-MCP distance.

    Returns:
        (21, 3) processed keypoints.
    """
    kp = kp - kp[0:1, :]
    dist = np.linalg.norm(kp[9])
    if dist < 1e-6:
        return kp
    scale = reference_wrist_to_mid_mcp / dist
    kp = kp * scale
    if correct_segments:
        kp = correct_segment_lengths(kp)
    kp[:, 2] *= depth_scale
    return kp


def draw_skeleton(frame: np.ndarray, raw_lm, connections=None, color=(0, 255, 0)):
    """Draw hand skeleton on a frame.

    Args:
        frame: BGR image (modified in place).
        raw_lm: List of (x, y) normalized landmark coordinates.
        connections: List of (i, j) index pairs (default: HAND_CONNECTIONS).
        color: Line color in BGR.
    """
    if connections is None:
        connections = HAND_CONNECTIONS
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y in raw_lm]
    for s, e in connections:
        cv2.line(frame, pts[s], pts[e], color, 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)

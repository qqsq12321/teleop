"""Teleoperation with MuJoCo Simulation.

Uses the Retargeter interface to map hand tracking input to robot hand joint
angles, visualized in MuJoCo simulation. Supports multiple robot hands.
"""

import argparse
import os
import pickle
import sys
import threading
import time
from pathlib import Path

if os.environ.get("MUJOCO_GL") is None and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anydexretarget import Retargeter
from input.camera import Camera
from input.mediapipe_replay import MediaPipeReplay
from input.quest3 import Quest3
from input.realsense import Realsense
from input.video import Video
from input.visionpro import VisionPro


ROBOT_HAND_CONFIGS = {
    "shadow_hand": {
        "model_path": lambda side: str(PROJECT_ROOT / "assets" / "shadow_hand" / f"scene_{side}.xml"),
        "needs_menagerie_mapping": True,
    },
    "wuji_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "wuji_hand" / "right.xml"),
    },
    "allegro_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "allegro_hand" / "scene_right.xml"),
        "qpos_mapping": [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7],
    },
    "inspire_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "inspire_hand" / "inspire_hand_right_mujoco.xml"),
        "qpos_mapping": [8, 9, 10, 11, 0, 1, 2, 3, 6, 7, 4, 5],
    },
    "ability_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "ability_hand" / "ability_hand_right_mujoco.xml"),
        "qpos_mapping": [8, 9, 0, 1, 2, 3, 6, 7, 4, 5],
    },
    "leap_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "leap_hand" / "leap_hand_right_mujoco.xml"),
        "qpos_mapping": [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7],
    },
    "svh_hand": {
        "model_path": lambda _: str(PROJECT_ROOT / "assets" / "schunk_hand" / "schunk_svh_hand_right_mujoco.xml"),
        "qpos_mapping": [0, 1, 2, 3, 8, 13, 14, 15, 16, 9, 10, 11, 12, 4, 5, 6, 7, 17, 18, 19],
    },
    "linkerhand_l21": {
        "model_path": lambda side: str(PROJECT_ROOT / "assets" / "linkerhand_l21" / f"linkerhand_l21_{side}_mujoco.xml"),
        "qpos_mapping": [0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8, 12, 13, 14, 15, 16],
        "qpos_servo_alpha": 0.2,
    },
    "rohand": {
        "model_path": lambda side: str(PROJECT_ROOT / "assets" / "rohand" / f"rohand_{side}_mujoco.xml"),
        "qpos_mapping": [3, 4, 1, 2, 0, 13, 14, 11, 12, 10, 18, 19, 16, 17, 15, 8, 9, 6, 7, 5, 20, 21, 23, 24, 22],
        "qpos_servo_alpha": 0.18,
        "base_quat": (0.7071068, 0, 0.7071068, 0),
    },
    "unitree_dex5_hand": {
        "model_path": lambda side: str(PROJECT_ROOT / "assets" / "unitree_dex5_hand" / f"unitree_dex5_hand_{side}_mujoco.xml"),
        "qpos_mapping": [16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "qpos_servo_alpha": 0.2,
        "base_quat": (0.7071068, 0.7071068, 0, 0),
    },
}


def map_urdf_to_mujoco_menagerie(qpos: np.ndarray) -> np.ndarray:
    """Map URDF joint angles (22 DoF) to MuJoCo Menagerie actuators (20 DoF)."""
    ctrl = np.zeros(20, dtype=np.float32)
    ctrl[0] = 0.0
    ctrl[1] = 0.0
    ctrl[2] = qpos[17]
    ctrl[3] = qpos[18]
    ctrl[4] = qpos[19]
    ctrl[5] = qpos[20]
    ctrl[6] = qpos[21]
    ctrl[7] = qpos[0]
    ctrl[8] = qpos[1]
    ctrl[9] = qpos[2] + qpos[3]
    ctrl[10] = qpos[9]
    ctrl[11] = qpos[10]
    ctrl[12] = qpos[11] + qpos[12]
    ctrl[13] = qpos[13]
    ctrl[14] = qpos[14]
    ctrl[15] = qpos[15] + qpos[16]
    ctrl[16] = qpos[4]
    ctrl[17] = qpos[5]
    ctrl[18] = qpos[6]
    ctrl[19] = qpos[7] + qpos[8]
    return ctrl


def retarget_to_mujoco_target(
    fingers_data: dict,
    hand_side: str,
    retargeter: Retargeter,
    hand_cfg: dict,
    target_len: int,
):
    """Retarget hand tracking input and map to MuJoCo control target."""
    fingers_pose = fingers_data[f"{hand_side}_fingers"]
    if np.allclose(fingers_pose, 0):
        return None

    qpos = retargeter.retarget(fingers_pose)

    if hand_cfg.get("needs_menagerie_mapping"):
        target = map_urdf_to_mujoco_menagerie(qpos)
    elif "qpos_mapping" in hand_cfg:
        target = qpos[hand_cfg["qpos_mapping"]]
    else:
        target = qpos

    target = np.asarray(target, dtype=np.float32)
    if len(target) != target_len:
        buf = np.zeros(target_len, dtype=np.float32)
        n = min(len(target), target_len)
        buf[:n] = target[:n]
        target = buf
    return target


def apply_qpos_to_mujoco(model, data, qpos, hand_cfg):
    """Apply retarget output qpos to MuJoCo model and step simulation."""
    if hand_cfg.get("needs_menagerie_mapping"):
        ctrl = map_urdf_to_mujoco_menagerie(qpos)
    elif "qpos_mapping" in hand_cfg:
        ctrl = qpos[hand_cfg["qpos_mapping"]]
    else:
        ctrl = qpos
    ctrl = np.asarray(ctrl, dtype=np.float32)

    qpos_servo_alpha = hand_cfg.get("qpos_servo_alpha")
    if qpos_servo_alpha is not None:
        n = min(len(ctrl), model.nq)
        data.qpos[:n] = ctrl[:n]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
    elif model.nu > 0:
        n = min(len(ctrl), model.nu)
        data.ctrl[:n] = ctrl[:n]
        for _ in range(200):
            mujoco.mj_step(model, data)
    else:
        n = min(len(ctrl), model.nq)
        data.qpos[:n] = ctrl[:n]
        mujoco.mj_forward(model, data)


def _resolve_example_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _apply_camera_config(
    camera,
    cam_cfg: dict,
    azimuth: float | None = None,
    elevation: float | None = None,
    distance: float | None = None,
    lookat: list[float] | None = None,
):
    camera.azimuth = cam_cfg.get("azimuth", 135) if azimuth is None else azimuth
    camera.elevation = cam_cfg.get("elevation", -20) if elevation is None else elevation
    camera.distance = cam_cfg.get("distance", 0.5) if distance is None else distance
    camera.lookat[:] = cam_cfg.get("lookat", [0, 0, 0.05]) if lookat is None else lookat


def run_teleop(
    hand_side: str = "right",
    config_path: str = "config/mediapipe/mediapipe_shadow_hand.yaml",
    input_device_type: str = "video",
    mediapipe_replay_path: str = "",
    video_path: str = "data/right.mp4",
    visionpro_ip: str = "192.168.50.127",
    quest3_port: int = 9000,
    quest3_protocol: str = "udp",
    playback_speed: float = 1.0,
    playback_loop: bool = True,
    enable_recording: bool = False,
    show_video: bool = False,
    video_depth_scale: float = 1.25,
    headless: bool = False,
    output_video_path: str = "",
    output_qpos_path: str = "",
    render_width: int = 1280,
    render_height: int = 720,
    max_frames: int | None = None,
    camera_azimuth: float | None = None,
    camera_elevation: float | None = None,
    camera_distance: float | None = None,
):
    """Run teleoperation with MuJoCo simulation."""
    hand_side = hand_side.lower()
    assert hand_side in {"right", "left"}, "hand_side must be 'right' or 'left'"

    config_file = Path(__file__).parent / config_path
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    robot_type = config.get("robot", {}).get("type", "shadow_hand")
    if robot_type not in ROBOT_HAND_CONFIGS:
        raise ValueError(f"Unknown robot type: {robot_type}. Supported: {list(ROBOT_HAND_CONFIGS.keys())}")

    hand_cfg = ROBOT_HAND_CONFIGS[robot_type]
    model_path = Path(hand_cfg["model_path"](hand_side))
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model file not found: {model_path}")

    print(f"  Robot: {robot_type}")
    print(f"  Model: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    force_control_cfg = hand_cfg.get("force_control")
    qpos_servo_alpha = hand_cfg.get("qpos_servo_alpha")
    direct_qpos_mode = hand_cfg.get("direct_qpos", False)
    qpos_servo_mode = (qpos_servo_alpha is not None) and not direct_qpos_mode
    actuator_mode = (model.nu > 0) and not direct_qpos_mode and not qpos_servo_mode
    force_control_mode = (force_control_cfg is not None) and not actuator_mode and not direct_qpos_mode and not qpos_servo_mode
    target_len = model.nu if actuator_mode else model.nq

    max_offscreen_width = int(model.vis.global_.offwidth)
    max_offscreen_height = int(model.vis.global_.offheight)
    if render_width > max_offscreen_width or render_height > max_offscreen_height:
        print(f"  Clamp offscreen render size: {render_width}x{render_height} -> {min(render_width, max_offscreen_width)}x{min(render_height, max_offscreen_height)}")
        render_width = min(render_width, max_offscreen_width)
        render_height = min(render_height, max_offscreen_height)

    if actuator_mode:
        print("  Control mode: actuator position")
        for i in range(model.nu):
            if model.actuator_ctrllimited[i]:
                ctrl_range = model.actuator_ctrlrange[i]
                data.ctrl[i] = (ctrl_range[0] + ctrl_range[1]) / 2
            else:
                data.ctrl[i] = 0.0
        for _ in range(100):
            mujoco.mj_step(model, data)
    elif qpos_servo_mode:
        print("  Control mode: qpos servo")
        mujoco.mj_forward(model, data)
    elif force_control_mode:
        print("  Control mode: joint PD force")
        mujoco.mj_forward(model, data)
    else:
        print("  Control mode: direct qpos")
        direct_qpos_mode = True
        mujoco.mj_forward(model, data)

    cam_cfg = config.get("render", {}).get("camera", {})
    viewer = None
    if not headless:
        viewer = mujoco.viewer.launch_passive(model, data)
        _apply_camera_config(
            viewer.cam,
            cam_cfg,
            azimuth=camera_azimuth,
            elevation=camera_elevation,
            distance=camera_distance,
        )

    device_map = {
        "visionpro": lambda: VisionPro(ip=visionpro_ip),
        "quest3": lambda: Quest3(port=quest3_port, protocol=quest3_protocol),
        "mediapipe_replay": lambda: MediaPipeReplay(
            record_path=mediapipe_replay_path,
            playback_speed=playback_speed,
            loop=playback_loop,
        ),
        "camera": lambda: Camera(camera_id=0, show_preview=True),
        "realsense": lambda: Realsense(hand_side=hand_side, show_video=show_video),
        "video": lambda: Video(
            video_path=video_path,
            hand_side=hand_side,
            show_video=show_video,
            playback_speed=playback_speed,
            loop=playback_loop,
            depth_scale=video_depth_scale,
        ),
    }
    if input_device_type not in device_map:
        raise ValueError(f"Unknown input device type: {input_device_type}")
    if input_device_type == "mediapipe_replay" and not mediapipe_replay_path:
        raise ValueError("mediapipe_replay_path is required for mediapipe_replay mode")

    input_device = device_map[input_device_type]()
    retargeter = Retargeter.from_yaml(str(config_file), hand_side)

    if input_device_type in ("mediapipe_replay", "video") and enable_recording:
        print("Note: Recording disabled in replay/video mode")
        enable_recording = False

    output_video = _resolve_example_path(output_video_path)
    output_qpos = _resolve_example_path(output_qpos_path)
    trajectory_log = [] if output_qpos is not None else None

    renderer = None
    render_cam = None
    video_writer = None
    if headless or output_video is not None:
        renderer = mujoco.Renderer(model, render_height, render_width)
        render_cam = mujoco.MjvCamera()
        render_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        _apply_camera_config(
            render_cam,
            cam_cfg,
            azimuth=camera_azimuth,
            elevation=camera_elevation,
            distance=camera_distance,
        )
        if output_video is not None:
            fps = 30.0
            if input_device_type == "video":
                fps = max(1.0, getattr(input_device, "fps", 30.0) * playback_speed)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(output_video), fourcc, fps, (render_width, render_height))
            if not video_writer.isOpened():
                raise RuntimeError(f"Cannot open output video writer: {output_video}")

    input_data_log = [] if enable_recording else None
    start_time = time.time()
    synchronous_input = headless and input_device_type == "video"

    latest_target = np.zeros(target_len, dtype=np.float32)
    ctrl_lock = threading.Lock()
    ctrl_ready = False
    stop_event = threading.Event()
    input_frame_count = 0

    def maybe_log_input(fingers_data: dict):
        if enable_recording and input_data_log is not None:
            input_data_log.append({
                "t": time.time() - start_time,
                "left_fingers": fingers_data["left_fingers"].copy(),
                "right_fingers": fingers_data["right_fingers"].copy(),
            })

    def apply_target(target: np.ndarray | None):
        if target is None:
            return
        if direct_qpos_mode:
            data.qpos[:] = target
            mujoco.mj_forward(model, data)
        elif actuator_mode:
            data.ctrl[:] = target
        elif qpos_servo_mode or force_control_mode:
            latest_target[:] = target

    def input_thread_fn():
        nonlocal ctrl_ready, input_frame_count
        while not stop_event.is_set():
            try:
                fingers_data = input_device.get_fingers_data()
            except Exception:
                break
            target = retarget_to_mujoco_target(
                fingers_data=fingers_data,
                hand_side=hand_side,
                retargeter=retargeter,
                hand_cfg=hand_cfg,
                target_len=target_len,
            )
            if target is None:
                time.sleep(0.005)
                continue
            maybe_log_input(fingers_data)
            with ctrl_lock:
                latest_target[:] = target
                ctrl_ready = True
            input_frame_count += 1

    input_thread = None if synchronous_input else threading.Thread(target=input_thread_fn, daemon=True)

    try:
        print("Starting teleoperation...")
        print(f"  Config: {config_path}")
        print(f"  Hand: {hand_side}")
        print(f"  Input: {input_device_type}")
        print(f"  Headless: {'ON' if headless else 'OFF'}")
        if output_video is not None:
            print(f"  Save video: {output_video}")
        if output_qpos is not None:
            print(f"  Save qpos: {output_qpos}")
        print(f"  Recording: {'ON' if enable_recording else 'OFF'}")
        print("=" * 50)

        if input_thread is not None:
            input_thread.start()

        render_count = 0
        fps_start_time = time.time()
        sim_dt = model.opt.timestep
        if synchronous_input and not direct_qpos_mode:
            target_fps = max(1.0, getattr(input_device, "fps", 30.0) * playback_speed)
            n_substeps = max(1, int(round((1.0 / target_fps) / sim_dt)))
        else:
            n_substeps = 10
        render_interval = sim_dt * n_substeps

        while True:
            loop_start = time.time()

            if viewer is not None and not viewer.is_running():
                break

            if synchronous_input:
                fingers_data = input_device.get_fingers_data()
                target = retarget_to_mujoco_target(
                    fingers_data=fingers_data,
                    hand_side=hand_side,
                    retargeter=retargeter,
                    hand_cfg=hand_cfg,
                    target_len=target_len,
                )
                if getattr(input_device, "_finished", False) and target is None:
                    break
                maybe_log_input(fingers_data)
                if target is not None:
                    apply_target(target)
                    input_frame_count += 1
            else:
                with ctrl_lock:
                    if ctrl_ready:
                        apply_target(latest_target)

            if direct_qpos_mode:
                if not synchronous_input:
                    mujoco.mj_forward(model, data)
            elif actuator_mode:
                for _ in range(n_substeps):
                    mujoco.mj_step(model, data)
            elif qpos_servo_mode:
                data.qpos[:target_len] += float(qpos_servo_alpha) * (latest_target - data.qpos[:target_len])
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
            elif force_control_mode:
                kp = float(force_control_cfg.get("kp", 10.0))
                kd = float(force_control_cfg.get("kd", 1.0))
                max_force = float(force_control_cfg.get("max_force", 2.0))
                for _ in range(n_substeps):
                    q_err = latest_target - data.qpos[:target_len]
                    tau = kp * q_err - kd * data.qvel[:target_len]
                    tau = np.clip(tau, -max_force, max_force)
                    data.qfrc_applied[:] = 0.0
                    data.qfrc_applied[:target_len] = tau
                    mujoco.mj_step(model, data)
            else:
                for _ in range(n_substeps):
                    mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()

            if trajectory_log is not None:
                trajectory_log.append({
                    "t": time.time() - start_time,
                    "target": np.array(data.qpos if direct_qpos_mode else (latest_target if (qpos_servo_mode or force_control_mode) else data.ctrl), copy=True),
                    "sim_qpos": np.array(data.qpos, copy=True),
                    "sim_ctrl": np.array(data.ctrl, copy=True) if model.nu > 0 else np.zeros(0, dtype=np.float32),
                })

            if renderer is not None and video_writer is not None:
                renderer.update_scene(data, camera=render_cam)
                frame_rgb = renderer.render()
                video_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            render_count += 1
            if render_count % 200 == 0:
                elapsed = time.time() - fps_start_time
                render_fps = render_count / elapsed
                print(f"Render FPS: {render_fps:.1f}  |  Input FPS: {input_frame_count / elapsed:.1f}")

            if max_frames is not None and render_count >= max_frames:
                break

            if not headless:
                elapsed_this_frame = time.time() - loop_start
                sleep_time = render_interval - elapsed_this_frame
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if (
                not synchronous_input
                and input_device_type in ("video", "mediapipe_replay")
                and not playback_loop
                and getattr(input_device, "_finished", False)
            ):
                break

    except KeyboardInterrupt:
        print("\nStopping controller...")
    finally:
        stop_event.set()
        if input_thread is not None:
            input_thread.join(timeout=2.0)
        if viewer is not None:
            viewer.close()
        if video_writer is not None:
            video_writer.release()
            print(f"Saved simulation video to {output_video}")
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass
        if trajectory_log is not None and output_qpos is not None:
            with open(output_qpos, "wb") as f:
                pickle.dump(trajectory_log, f)
            print(f"Saved trajectory log with {len(trajectory_log)} frames to {output_qpos}")

    return input_data_log


def main():
    parser = argparse.ArgumentParser(
        description="Teleoperation with MuJoCo Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file (overrides --robot)")
    parser.add_argument("--robot", type=str, default="shadow",
                        choices=["shadow", "wuji", "allegro", "leap",
                                 "inspire", "ability", "svh", "rohand",
                                 "linkerhand_l21", "unitree_dex5"],
                        help="Robot hand type (default: shadow)")
    parser.add_argument("--hand", type=str, default="right", choices=["left", "right"],
                        help="Hand side (default: right)")

    parser.add_argument("--input", type=str, default=None,
                        choices=["visionpro", "quest3", "mediapipe_replay", "camera", "realsense", "video"],
                        help="Input device type")
    parser.add_argument("--realsense", action="store_true",
                        help="Use RealSense camera (shortcut for --input realsense)")
    parser.add_argument("--show-video", action="store_true",
                        help="Show video with MediaPipe landmarks overlay")
    parser.add_argument("--video-depth-scale", type=float, default=1.25,
                        help="Extra scale applied to MediaPipe z/depth for video input (default: 1.25)")

    parser.add_argument("--play", type=str, default=None, metavar="FILE",
                        help="Play MediaPipe recording file (shortcut for --input mediapipe_replay)")
    parser.add_argument("--video", type=str, default=None, metavar="FILE",
                        help="Play MP4/AVI video file with MediaPipe detection (shortcut for --input video)")

    parser.add_argument("--ip", type=str, default="192.168.50.127",
                        help="VisionPro IP address (default: 192.168.50.127)")
    parser.add_argument("--port", type=int, default=9000,
                        help="Quest 3 HTS listener port (default: 9000)")
    parser.add_argument("--protocol", type=str, default="udp", choices=["udp", "tcp"],
                        help="Quest 3 HTS transport protocol (default: udp)")

    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed for replay mode (default: 1.0)")
    parser.add_argument("--no-loop", action="store_true",
                        help="Disable looping for replay/video mode")

    parser.add_argument("--record", action="store_true",
                        help="Record input data to file")
    parser.add_argument("--output", type=str, default=None, metavar="FILE",
                        help="Output file for recording (default: auto-generated)")

    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI viewer")
    parser.add_argument("--save-sim", type=str, default=None, metavar="FILE",
                        help="Save MuJoCo simulation video to MP4")
    parser.add_argument("--save-qpos", type=str, default=None, metavar="FILE",
                        help="Save target/sim qpos trajectory to PKL")
    parser.add_argument("--width", type=int, default=1280,
                        help="Offscreen render width (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="Offscreen render height (default: 720)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of simulation/render frames")
    parser.add_argument("--cam-azimuth", type=float, default=None,
                        help="Override camera azimuth")
    parser.add_argument("--cam-elevation", type=float, default=None,
                        help="Override camera elevation")
    parser.add_argument("--cam-distance", type=float, default=None,
                        help="Override camera distance")

    args = parser.parse_args()

    input_device_type = args.input
    mediapipe_replay_path = ""
    video_path = ""

    if args.realsense:
        input_device_type = "realsense"
    elif args.video:
        input_device_type = "video"
        video_path = args.video
    elif args.play:
        input_device_type = "mediapipe_replay"
        mediapipe_replay_path = args.play

    if input_device_type is None:
        input_device_type = "video"
        video_path = "data/right.mp4"

    if input_device_type == "mediapipe_replay" and not mediapipe_replay_path:
        parser.error("--play FILE is required for mediapipe_replay mode")
    if input_device_type == "video" and not video_path:
        parser.error("--video FILE is required for video mode")

    config_path = args.config
    if config_path is None:
        robot_name_map = {
            "shadow": "shadow_hand", "wuji": "wuji_hand", "allegro": "allegro_hand",
            "leap": "leap_hand", "inspire": "inspire_hand", "ability": "ability_hand",
            "svh": "svh_hand", "rohand": "rohand", "linkerhand_l21": "linkerhand_l21",
            "unitree_dex5": "unitree_dex5_hand",
        }
        input_to_dir = {
            "quest3": "quest3",
            "visionpro": "avp",
        }
        config_dir = input_to_dir.get(input_device_type, "mediapipe")
        robot_file = robot_name_map.get(args.robot, args.robot)
        config_path = f"config/{config_dir}/{config_dir}_{robot_file}.yaml"

    log = run_teleop(
        hand_side=args.hand,
        config_path=config_path,
        input_device_type=input_device_type,
        mediapipe_replay_path=mediapipe_replay_path,
        video_path=video_path,
        visionpro_ip=args.ip,
        quest3_port=args.port,
        quest3_protocol=args.protocol,
        playback_speed=args.speed,
        playback_loop=not args.no_loop,
        enable_recording=args.record,
        show_video=args.show_video,
        video_depth_scale=args.video_depth_scale,
        headless=args.headless or args.save_sim is not None,
        output_video_path=args.save_sim or "",
        output_qpos_path=args.save_qpos or "",
        render_width=args.width,
        render_height=args.height,
        max_frames=args.max_frames,
        camera_azimuth=args.cam_azimuth,
        camera_elevation=args.cam_elevation,
        camera_distance=args.cam_distance,
    )

    if log is not None and len(log) > 0:
        if args.output:
            log_path = Path(args.output)
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            log_path = Path(__file__).parent / f"input_data_log_{timestamp}.pkl"
        with open(log_path, "wb") as f:
            pickle.dump(log, f)
        print(f"Saved input data log with {len(log)} entries to {log_path}")


if __name__ == "__main__":
    main()

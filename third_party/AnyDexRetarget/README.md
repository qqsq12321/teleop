[中文](README.zh.md) | English

# AnyDexRetarget

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

High-precision hand pose retargeting system. Based on adaptive analytical optimization, with support for multiple dexterous hands and multiple hand-tracking input sources for simulation and teleoperation.

## Demo

### Simulation Retargeting

https://github.com/user-attachments/assets/0950b2b0-ecd4-4270-abf6-5729dc05c6cb

### Quest 3 Hand-Arm Teleoperation

https://github.com/user-attachments/assets/4bcac46b-a603-4c0c-9d70-83d4351c9811

## Features

- **Shadow Hand Support**: Shadow Hand with MuJoCo Menagerie high-quality meshes
- **High-Precision Pinch**: Adaptive optimization for accurate finger-to-thumb contact
- **Real-time Performance**: Analytical gradients + NLopt SLSQP (~2ms per frame)
- **Multiple Input Sources**: Apple Vision Pro, Meta Quest 3, laptop camera (MediaPipe), recorded data replay

## Table of Contents

- [Supported Robots](#supported-robots)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Optimizer Details](#optimizer-details)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [Contact](#contact)

## Supported Robots

Config files are grouped by input source:

- `example/config/mediapipe/mediapipe_<robot>.yaml` for camera / video / replay input
- `example/config/avp/avp_<robot>.yaml` for Apple Vision Pro input
- `example/config/quest3/quest3_<robot>.yaml` for Meta Quest 3 input

| Robot | `--robot` value | Config suffix | Description |
|-------|------------------|---------------|-------------|
| **Shadow Hand** | `shadow` | `shadow_hand` | Shadow Hand with MuJoCo Menagerie meshes (default sim target) |
| **Wuji Hand** | `wuji` | `wuji_hand` | Wuji Hand, 5 fingers / 20 DOF |
| **Allegro Hand** | `allegro` | `allegro_hand` | Allegro Hand, 4 fingers / 16 DOF |
| **Inspire Hand** | `inspire` | `inspire_hand` | Inspire Hand with mimic joints |
| **Ability Hand** | `ability` | `ability_hand` | Ability Hand with mimic joints |
| **Leap Hand** | `leap` | `leap_hand` | Leap Hand, 4 fingers / 16 DOF |
| **SVH Hand** | `svh` | `svh_hand` | Schunk SVH Hand with mimic joints |
| **LinkerHand L21** | `linkerhand_l21` | `linkerhand_l21` | LinkerHand L21 |
| **ROHand** | `rohand` | `rohand` | ROHand |
| **Unitree Dex5** | `unitree_dex5` | `unitree_dex5_hand` | Unitree Dex5 |

## Repository Structure

```text
├── anydexretarget/
│   ├── retarget.py                        # High-level unified interface
│   ├── robot.py                           # Pinocchio robot wrapper
│   ├── mediapipe.py                       # MediaPipe coordinate transforms
│   └── optimizer/                         # Optimizer implementations
│       ├── base_optimizer.py              # Base optimizer with FK/Jacobian
│       ├── analytical_optimizer.py        # Adaptive optimizer with analytical gradients
│       ├── robot_configs.py               # Robot link/URDF configurations
│       └── utils.py                       # TimingStats, LPFilter, Huber loss
├── example/
│   ├── teleop_sim.py                      # MuJoCo simulation demo
│   ├── teleop_real.py                     # Real hardware control
│   ├── input/                             # Input device modules
│   │   ├── landmark_utils.py              # Shared MediaPipe landmark processing
│   │   ├── camera.py / video.py / ...     # Input devices
│   ├── test/                              # Debug & visualization tools
│   ├── config/                            # YAML configurations (by input source)
│   │   ├── avp/                           # Apple Vision Pro configs
│   │   ├── quest3/                        # Meta Quest 3 configs
│   │   └── mediapipe/                     # MediaPipe (camera/video/replay) configs
│   └── data/                              # Sample recordings
├── assets/                                # Robot URDF / MuJoCo assets
└── requirements.txt
```

## Installation

### Prerequisites

- Python >= 3.10
- (Optional) Apple Vision Pro with [Tracking Streamer](https://apps.apple.com/us/app/tracking-streamer/id6478969032) app
- (Optional) Meta Quest 3 with [Hand Tracking Streamer](https://github.com/wengmister/hand-tracking-streamer) app

### Install

```bash
git clone https://gitee.com/gx_robot/AnyDexRetarget.git
cd AnyDexRetarget

# Install pinocchio via conda (recommended, pre-built binaries)
conda install -c conda-forge pinocchio

# Install other dependencies
pip install -r requirements.txt
pip install -e .
```

### Troubleshooting

**pinocchio Installation**: `pinocchio` must be installed via conda (not pip). The pip package `pin` requires C++ compilation and often fails on Windows:
```bash
conda install -c conda-forge pinocchio
```

**macOS MuJoCo**: Use `mjpython` instead of `python`:
```bash
mjpython example/teleop_sim.py --video example/data/right.mp4
```

## Quick Start

The repository currently includes:

- `example/data/right.mp4`: sample input video
- `example/data/avp1.pkl`: optional recorded hand-tracking replay

### Simulation

```bash
cd example

# Run the included sample video
python teleop_sim.py --video data/right.mp4 --robot shadow --hand right

# Replay the optional sample recording
python teleop_sim.py --play data/avp1.pkl --robot shadow --hand right

# Real-time with laptop camera (MediaPipe)
python teleop_sim.py --input camera --robot shadow --hand right

# Real-time with Vision Pro
python teleop_sim.py --input visionpro --robot shadow --ip <vision-pro-ip> --hand right

# Real-time with Quest 3 (via Hand Tracking Streamer)
python teleop_sim.py --input quest3 --robot shadow --port 9000 --hand right

# Real-time with RealSense
python teleop_sim.py --realsense --robot shadow --hand right --show-video

# Replay your own recording (.pkl)
python teleop_sim.py --play path/to/record.pkl --robot shadow --hand right
```

### Real Hardware

`teleop_real.py` demonstrates real hardware teleoperation using **Wuji Hand** as an example. It sends `5 x 4` joint targets through `wujihandpy`. You can adapt the control loop for other robot hands. It supports `visionpro` and `mediapipe_replay` input.

```bash
cd example

# Live Vision Pro -> Wuji Hand
python teleop_real.py --robot wuji --input visionpro --ip <vision-pro-ip> --hand right

# Replay the optional sample recording -> Wuji Hand
python teleop_real.py --robot wuji --play data/avp1.pkl --hand right

# Replay your own recording -> Wuji Hand
python teleop_real.py --robot wuji --play path/to/record.pkl --hand right

# Linux USB permission
sudo chmod a+rw /dev/ttyUSB0
```

### Command Reference

| Option | Default | Description |
|--------|---------|-------------|
| `--robot` | `shadow` (sim) / `wuji` (real) | Robot hand type |
| `--config` | auto-select | Configuration file (overrides `--robot`) |
| `--hand` | `right` | Hand side (`left`/`right`) |
| `--input` | - | `teleop_sim.py`: `visionpro` / `quest3` / `camera` / `realsense` / `video` / `mediapipe_replay` |
| `--input` | - | `teleop_real.py`: `visionpro` / `mediapipe_replay` |
| `--realsense` | off | Shortcut for `--input realsense` |
| `--play FILE` | - | Replay recording (shortcut for `--input mediapipe_replay`) |
| `--video FILE` | - | Video file input with MediaPipe hand detection |
| `--ip` | `192.168.50.127` | Vision Pro IP |
| `--port` | `9000` | Quest 3 HTS listener port |
| `--protocol` | `udp` | Quest 3 HTS transport protocol (`udp`/`tcp`) |
| `--speed` | `1.0` | Playback speed |
| `--record` | - | Record input data |
| `--output FILE` | - | Output file path for recording |
| `--show-video` | off | Show RGB / landmark preview for supported inputs |
| `--video-depth-scale` | `1.25` | Extra depth scaling for `--video` mode |
| `--no-loop` | - | Disable looping for replay |
| `--headless` | off | Run simulation without GUI viewer |
| `--save-sim FILE` | - | Save offscreen simulation video |
| `--save-qpos FILE` | - | Save target / simulated qpos trajectory |

### Debug & Visualization Tools

#### debug_skeleton.py

Compare three hand skeletons in the MuJoCo viewer to debug retargeting issues:

- **Blue**: Raw MediaPipe skeleton (after coordinate transform, before scaling)
- **Green**: Scaled target skeleton (what the optimizer tries to match)
- **Red**: Robot FK skeleton (retargeting result)

```bash
cd example

# With camera input
python test/debug_skeleton.py --robot leap --input camera

# With video file
python test/debug_skeleton.py --robot leap --video data/right.mp4

# With optional sample recording
python test/debug_skeleton.py --robot shadow --play data/avp1.pkl

# With your own recorded data
python test/debug_skeleton.py --robot shadow --play path/to/record.pkl
```

#### visualize_scaling.py

Visualize how `scaling` and `segment_scaling` parameters affect MediaPipe keypoints. Shows the original skeleton vs the scaled target skeleton in a matplotlib 3D plot.

```bash
cd example

# With video file
python test/visualize_scaling.py --robot leap --video data/right.mp4 --hand right

# With optional sample recording
python test/visualize_scaling.py --robot allegro --play data/avp1.pkl --hand right

# With your own recorded data
python test/visualize_scaling.py --robot allegro --play path/to/record.pkl --hand right
```

## Configuration

### Config File Structure

```yaml
optimizer:
  type: "AdaptiveOptimizerAnalytical"

robot:
  type: "shadow_hand"

retarget:
  # Loss weights
  w_pos: 1.0              # Tip position weight
  w_dir: 5.0              # Tip direction weight
  w_full_hand: 1.0        # Full hand weight

  # Huber loss thresholds
  huber_delta: 2.0        # Position threshold (cm)
  huber_delta_dir: 0.5    # Direction threshold

  # Regularization
  norm_delta: 0.04        # Velocity smoothing

  # Scaling
  scaling: 0.81           # MediaPipe to robot scale (Shadow Hand ~81% of MediaPipe)

  # Coordinate alignment
  mediapipe_rotation:
    x: 0.0
    y: 0.0
    z: -90.0              # Shadow Hand requires -90° Z rotation

  # Pinch thresholds (cm)
  pinch_thresholds:
    index:  { d1: 2.0, d2: 4.0 }
    middle: { d1: 2.0, d2: 4.0 }
    ring:   { d1: 2.0, d2: 4.0 }
    pinky:  { d1: 2.0, d2: 4.0 }

  # Low-pass filter (0~1, smaller = smoother)
  lp_alpha: 0.4
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `scaling` | Hand size ratio. Shadow Hand ≈ 0.81 |
| `mediapipe_rotation.z` | Coordinate alignment. Shadow Hand = -90° |

## API Reference

### Basic Usage

```python
from anydexretarget import Retargeter

# Load from config file
retargeter = Retargeter.from_yaml("config/mediapipe/mediapipe_shadow_hand.yaml", hand_side="right")

# Retarget: (21, 3) MediaPipe keypoints -> joint angles
qpos = retargeter.retarget(raw_keypoints)

# With verbose output
qpos, info = retargeter.retarget_verbose(raw_keypoints)
print(f"Cost: {info['cost']:.4f}")
print(f"Pinch alphas: {info['pinch_alphas']}")
```

### Advanced Usage

```python
# Direct optimizer access
optimizer = retargeter.optimizer

# Compute cost for given pose
cost = optimizer.compute_cost(qpos, mediapipe_keypoints)

# Get timing statistics
stats = optimizer.get_timing_stats()
print(f"Average time: {stats.avg_total_ms:.2f} ms")
```

## Optimizer Details

### Optimization Formulation

```
min_q  L(q) + λ||q - q_prev||²
s.t.   q_min ≤ q ≤ q_max
```

### Loss Function

```
L = Σᵢ [αᵢ · L_tip_dir_vec + (1-αᵢ) · L_full_hand] + norm_delta · ||Δq||²
```

- **L_tip_dir_vec**: Position + direction matching (for pinch gestures)
- **L_full_hand**: Full hand vector matching (for open hand)

### Adaptive Blending

```
αᵢ = 0.7    if dᵢ < d1  (pinching → TipDirVec mode)
αᵢ = 0.0    if dᵢ > d2  (open → FullHandVec mode)
αᵢ = lerp   otherwise
```

Where `dᵢ` is thumb-to-finger distance.

## Citation

```bibtex
@software{anydexretarget2025,
  title={AnyDexRetarget},
  author={Shiquan Qiu},
  year={2025},
  url={https://gitee.com/gx_robot/AnyDexRetarget},
}
```

## Acknowledgement

- [MuJoCo](https://mujoco.org/) - Physics simulation
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) - Shadow Hand models
- [dex-retargeting](https://github.com/dexsuite/dex-retargeting) - Retargeting algorithms
- [DexPilot](https://arxiv.org/abs/1910.03135) - Vision-based teleoperation
- [VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop) - Apple Vision Pro streaming
- [wuji-retargeting](https://github.com/wuji-technology/wuji-retargeting) - Wuji retargeting

## Contact

For questions, please open an issue on [Gitee](https://gitee.com/gx_robot/AnyDexRetarget/issues) or contact the author via 932851972@qq.com.

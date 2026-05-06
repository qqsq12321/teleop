"""Interactive viewer starting at rm65 home_qpos, prints joint_1..6 live.

Use the viewer's left-panel Control/Joint sliders to pose the arm, then
read joint_1..6 values either from the viewer's Joint sidebar or from the
periodic stdout prints below.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "third_party" / "AnyDexRetarget"),
)

import importlib.util

spec = importlib.util.spec_from_file_location(
    "teleop_sim",
    str(Path(__file__).resolve().parent.parent / "example" / "teleop_sim.py"),
)
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)

import mujoco
import numpy as np
from mujoco import viewer as V

cfg = ts.ROBOT_CONFIGS["rm65"]
home = cfg["home_qpos"]

m = mujoco.MjModel.from_xml_path(cfg["scene_xml"])
d = mujoco.MjData(m)
n = min(m.nq, home.shape[0])
d.qpos[:n] = home[:n]
d.ctrl[:n] = home[:n]
mujoco.mj_forward(m, d)

print(f"Starting at home_qpos = {home.tolist()}")
print("Drag the 'Control' or 'Joint' sliders in the viewer's left sidebar.")
print("Current joint_1..6 is printed here every second.")
print()

site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, cfg["site_name"])
last_print = 0.0

with V.launch_passive(m, d) as vis:
    while vis.is_running():
        mujoco.mj_step(m, d)
        vis.sync()
        now = time.time()
        if now - last_print > 1.0:
            q = d.qpos[:6].copy()
            ee = d.site_xpos[site_id].copy()
            print(
                f"qpos = [{q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f}, "
                f"{q[3]:+.4f}, {q[4]:+.4f}, {q[5]:+.4f}]  "
                f"EE=({ee[0]:+.3f}, {ee[1]:+.3f}, {ee[2]:+.3f})"
            )
            last_print = now
        sleep_time = m.opt.timestep - (time.time() - now + 1e-4)
        if sleep_time > 0:
            time.sleep(sleep_time)

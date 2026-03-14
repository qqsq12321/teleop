# Quest Teleop Mujoco

Mujoco teleop demo setup for the Quest [Hand Tracking Streamer](https://github.com/wengmister/quest-wrist-tracker) App

# 运行命令：
PYTHONPATH=/home/hand PYTHONUNBUFFERED=1 python3 teleop_env/teleop_kinova_wuji_real.py --kinova-ip 192.168.1.10 --port 9000 --move-home-on-start --arm-max-speed-deg 50 --arm-kp 2.0 --position-scale 1.0 --wrist-pos-deadband 0.03 --enable-wrist-rotation --disable-hand
    
    ############
      --move-home-on-start \          # 启动时先让 Kinova 回到保存的 Home 位
  --arm-max-speed-deg 50 \        # 每个关节的最大速度上限：50 deg/s
  --arm-kp 2.0 \                  # 关节误差转速度的比例系数，越大追得越积极
  --position-scale 1.0 \          # 手腕位移映射倍率，1.0 表示手移多少机械臂末端大致跟多少
  --wrist-pos-deadband 0.03 \     # 手腕位置死区：3 cm 内的小抖动忽略
  --enable-wrist-rotation \       # 开启 wrist 旋转跟随
  --disable-hand                  # 禁用 Wuji 手，只测试机械臂


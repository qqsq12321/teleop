"""Quick test: open and close Robotiq gripper via Kortex API."""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

_KORTEX_EXAMPLES_DIR = (
    Path(__file__).resolve().parent
    / "Kinova-kortex2_Gen3_G3L"
    / "api_python"
    / "examples"
)
sys.path.insert(0, str(_KORTEX_EXAMPLES_DIR))

import utilities as kortex_utilities
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2


def main():
    kinova_args = SimpleNamespace(ip="192.168.1.10", username="admin", password="admin")

    with kortex_utilities.DeviceConnection.createTcpConnection(kinova_args) as router:
        base = BaseClient(router)

        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger = gripper_command.gripper.finger.add()
        finger.finger_identifier = 1

        print("Closing gripper to 0.5...")
        finger.value = 0.5
        base.SendGripperCommand(gripper_command)
        time.sleep(2)

        print("Opening gripper to 0.0...")
        finger.value = 0.0
        base.SendGripperCommand(gripper_command)
        time.sleep(2)

        print("Closing gripper to 1.0...")
        finger.value = 1.0
        base.SendGripperCommand(gripper_command)
        time.sleep(2)

        print("Opening gripper to 0.0...")
        finger.value = 0.0
        base.SendGripperCommand(gripper_command)
        time.sleep(1)

        print("Done.")


if __name__ == "__main__":
    main()

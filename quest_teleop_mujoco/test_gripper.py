"""Test gripper open/close independently."""
import sys
from pathlib import Path
from types import SimpleNamespace
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "Kinova-kortex2_Gen3_G3L" / "api_python" / "examples"))

import utilities as kortex_utilities
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2

def main():
    args = SimpleNamespace(ip="192.168.1.10", username="admin", password="admin")

    with kortex_utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)

        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger = gripper_command.gripper.finger.add()
        finger.finger_identifier = 0

        print("Closing gripper (position=1.0)...")
        finger.value = 1.0
        base.SendGripperCommand(gripper_command)
        time.sleep(3)

        print("Opening gripper (position=0.0)...")
        finger.value = 0.0
        base.SendGripperCommand(gripper_command)
        time.sleep(3)

        print("Done.")

if __name__ == "__main__":
    main()

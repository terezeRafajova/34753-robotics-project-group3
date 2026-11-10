"""Ping IDs 0-20 and list which motors answer."""
import glob
from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

port = PortHandler(glob.glob('/dev/tty.usbmodem*')[0])
packet = PacketHandler(1.0)
port.openPort()
port.setBaudRate(1_000_000)

found = []
for dxl_id in range(21):
    model, result, error = packet.ping(port, dxl_id)
    if result == COMM_SUCCESS:
        found.append(dxl_id)
        print(f"ID {dxl_id}: found (model {model})")

print("Motors found:", found if found else "none")
port.closePort()

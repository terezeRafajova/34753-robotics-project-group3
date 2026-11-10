"""Move the 4 arm joints to (0, 0, 0, 0) degrees, slowly.

Uses the same setup as MyRobot.py: AX-12A servos, Protocol 1.0,
1 Mbaud, IDs 1-4, and a 150 deg offset so 0 deg = middle of the servo range.
"""
import glob
import time

from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

# --- Settings ---------------------------------------------------------------
MOTOR_IDS = [1, 2, 3, 4]
TARGET_DEG = [0, 0, 0, 0]
OFFSET_DEG = 150             # servo position (deg) that counts as joint 0
SPEED = 0.1                  # 0..1, keep it slow for testing
BAUDRATE = 1_000_000
PROTOCOL_VERSION = 1.0

# AX-12A control table
ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_POSITION = 30
ADDR_MOVING_SPEED = 32
ADDR_PRESENT_POSITION = 36
DEG_PER_TICK = 0.29


def find_port():
    ports = glob.glob('/dev/tty.usbmodem*')
    if not ports:
        raise SystemExit("No robot found. Is the USB2AX plugged in?")
    return ports[0]


def check(result, error, what):
    if result != COMM_SUCCESS:
        raise IOError(f"{what}: {packet.getTxRxResult(result)}")
    if error != 0:
        raise IOError(f"{what}: {packet.getRxPacketError(error)}")


port = PortHandler(find_port())
packet = PacketHandler(PROTOCOL_VERSION)

if not port.openPort():
    raise SystemExit("Failed to open the port.")
if not port.setBaudRate(BAUDRATE):
    raise SystemExit("Failed to set the baud rate.")
print(f"Connected on {port.getPortName()}")

try:
    # Set a slow speed, enable torque and send the goal to each joint
    for dxl_id, deg in zip(MOTOR_IDS, TARGET_DEG):
        ticks = round((deg + OFFSET_DEG) / DEG_PER_TICK)
        check(*packet.write2ByteTxRx(port, dxl_id, ADDR_MOVING_SPEED, round(SPEED * 1023)), f"speed ID {dxl_id}")
        check(*packet.write1ByteTxRx(port, dxl_id, ADDR_TORQUE_ENABLE, 1), f"torque ID {dxl_id}")
        check(*packet.write2ByteTxRx(port, dxl_id, ADDR_GOAL_POSITION, ticks), f"goal ID {dxl_id}")

    # Wait until every joint is within 2 deg of its target (max 10 s)
    start = time.time()
    while time.time() - start < 10:
        angles = []
        for dxl_id in MOTOR_IDS:
            ticks, result, error = packet.read2ByteTxRx(port, dxl_id, ADDR_PRESENT_POSITION)
            check(result, error, f"read ID {dxl_id}")
            angles.append(ticks * DEG_PER_TICK - OFFSET_DEG)
        if all(abs(a - t) < 2 for a, t in zip(angles, TARGET_DEG)):
            break
        time.sleep(0.05)

    print("Joint angles (deg):", [round(a, 1) for a in angles])

finally:
    # Torque stays on so the arm holds its position. Uncomment to let it go limp:
    # for dxl_id in MOTOR_IDS:
    #     packet.write1ByteTxRx(port, dxl_id, ADDR_TORQUE_ENABLE, 0)
    port.closePort()

"""AX-12A position control. The firmware drives each joint between waypoints."""
import glob
import time

from dynamixel_sdk import COMM_RX_TIMEOUT, COMM_SUCCESS, GroupSyncWrite, PacketHandler, PortHandler
import numpy as np

LIMITS = np.array([[-90, 90], [-120, 120], [-140, 140], [-140, 140]])
IDS = (1, 2, 3, 4)

# AX-12 compliance slope (registers 28/29): how softly a motor approaches its goal.
# Factory value 32 lets a loaded joint stop ~2 degrees short (the shoulder sags when the
# arm reaches low). Smaller = stiffer. Allowed: 2, 4, 8, 16, 32, 64, 128. If a joint
# starts to buzz or oscillate, go back up one step.
COMPLIANCE_SLOPE = {1: 32, 2: 32, 3: 32, 4: 32}   # factory values (16 made base + shoulder oscillate)


def check_angles(q):
    if np.shape(q) != (4,) or not np.all((q >= LIMITS[:, 0]) & (q <= LIMITS[:, 1])):
        raise ValueError(f"Joint target outside the arm workspace: {q}")


def find_port():
    """First USB2AX on a Mac (/dev/tty.usbmodem*)."""
    ports = sorted(glob.glob("/dev/tty.usbmodem*"))
    if not ports:
        raise OSError("No robot found. Is the USB2AX plugged in? (or pass the port, e.g. Arm('COM6'))")
    return ports[0]


class Arm:
    def __init__(self, device=None):
        device = device or find_port()
        self.port = PortHandler(device)
        self.packet = PacketHandler(1.0)
        try:
            if not self.port.setBaudRate(1_000_000):
                raise OSError(f"Cannot open {device}.")
            self.limits = []
            for motor in IDS:
                model = self.read_register(motor, 0)
                low, high = self.read_register(motor, 6), self.read_register(motor, 8)
                if model != 12 or not 0 <= low < high <= 1023 or not self.read_register(motor, 34):
                    raise OSError(f"ID {motor}: expected AX-12 position mode with available torque.")
                self.limits.append((low, high))
            positions = self.hold()
            for motor, tick in zip(IDS, self.validate(positions)):
                if self.read_register(motor, 30) != tick:
                    raise OSError(f"ID {motor} did not accept its initial hold position.")
            for motor in IDS:
                for register in (28, 29):          # CW and CCW compliance slope
                    result, error = self.packet.write1ByteTxRx(self.port, motor, register,
                                                               COMPLIANCE_SLOPE[motor])
                    self.check(result, error)
                result, error = self.packet.write1ByteTxRx(self.port, motor, 24, 1)
                self.check(result, error)
        except BaseException:
            self.port.closePort()
            raise

    def check(self, result, error):
        if result != COMM_SUCCESS:
            raise OSError(self.packet.getTxRxResult(result))
        if error:
            raise OSError(self.packet.getRxPacketError(error))

    def read_register(self, motor, address):
        for _ in range(3):
            value, result, error = self.packet.read2ByteTxRx(self.port, motor, address)
            if result != COMM_RX_TIMEOUT:
                break
        try:
            self.check(result, error)
        except OSError as exc:
            raise OSError(f"ID {motor}, register {address}: {exc}") from exc
        return value

    def angles(self):
        return np.array([self.read_register(motor, 36) for motor in IDS]) * 0.29 - 150

    def validate(self, goal):
        check_angles(goal)
        ticks = np.rint((goal + 150) / 0.29).astype(int)
        for motor, tick, (low, high) in zip(IDS, ticks, self.limits):
            if not low <= tick <= high:
                raise ValueError(f"ID {motor}: {tick} outside programmed limits {low}..{high}.")
        return ticks

    def command(self, goal, speeds):
        packet = GroupSyncWrite(self.port, self.packet, 30, 4)
        for motor, tick, speed in zip(IDS, self.validate(goal), speeds):
            data = list(int(tick).to_bytes(2, "little") + int(speed).to_bytes(2, "little"))
            if not packet.addParam(motor, data):
                raise OSError(f"Cannot construct command for ID {motor}.")
        self.check(packet.txPacket(), 0)

    def hold(self):
        current = self.angles()
        self.command(current, [40] * 4)
        return current

    def move(self, goal, speed, monitor):
        """Send one coordinated goal. Read encoders without pausing the servos."""
        current = self.angles()
        delta = np.abs(goal - current)
        duration = max(float(delta.max()) / speed, 0.12)
        # One AX speed unit is 0.111 rpm, approximately 0.666 degrees/s.
        speeds = np.clip(np.rint(delta / duration / 0.666), 1, 1023).astype(int)
        if monitor(current):
            return current
        self.command(goal, speeds)
        deadline = time.monotonic() + duration * 2 + 1.2
        while True:
            current = self.angles()
            if monitor(current):
                return current
            if time.monotonic() > deadline:
                raise RuntimeError(f"Move timed out. Goal {goal.round(1)}, actual {current.round(1)}.")
            time.sleep(0.018)

    # ---- Simple helpers for the scripts in tests/ ------------------------------------
    def move_j(self, q, speed=30.0, tol=2.0):
        """Move to joint angles q [deg] at about `speed` deg/s and wait until within `tol`."""
        goal = np.asarray(q, dtype=float)
        return self.move(goal, speed, lambda current: np.max(np.abs(current - goal)) <= tol)

    def read(self):
        """Present joint angles [deg]."""
        return self.angles()

    def close(self):
        self.port.closePort()

"""Small driver for the AX-12A arm (IDs 1-4), joint angles in DH degrees.

q = 0 is the arm straight up (servo position 150 deg).
"""
import glob
import time

from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_POSITION = 30
ADDR_MOVING_SPEED = 32
ADDR_PRESENT_POSITION = 36
DEG_PER_TICK = 0.29


class Arm:
    MOTOR_IDS = [1, 2, 3, 4]
    OFFSET_DEG = [150, 150, 150, 150]   # servo angle that equals q = 0
    SIGN = [1, 1, 1, 1]                 
    LIMITS = [(-90, 90), (-120, 120), (-140, 140), (-140, 140)]

    def __init__(self, speed=0.1, baudrate=1_000_000):
        ports = glob.glob('/dev/tty.usbmodem*')
        if not ports:
            raise SystemExit("No robot found. Is the USB2AX plugged in?")
        self.port = PortHandler(ports[0])
        self.packet = PacketHandler(1.0)
        if not self.port.openPort() or not self.port.setBaudRate(baudrate):
            raise SystemExit("Could not open the port.")
        for i in self.MOTOR_IDS:
            self._write2(i, ADDR_MOVING_SPEED, round(speed * 1023))
            self._write1(i, ADDR_TORQUE_ENABLE, 1)

    # --- low level ---
    def _check(self, result, error, what):
        if result != COMM_SUCCESS:
            raise IOError(f"{what}: {self.packet.getTxRxResult(result)}")
        if error != 0:
            raise IOError(f"{what}: {self.packet.getRxPacketError(error)}")

    def _retry(self, fn, what, retries=5):
        """Call fn() until the motor answers; one lost packet is not a reason to crash."""
        for attempt in range(retries):
            out = fn()
            result, error = out[-2], out[-1]
            if result == COMM_SUCCESS and error == 0:
                return out
            time.sleep(0.02)
        self._check(result, error, what)        # still failing: raise a readable error

    def _write1(self, dxl_id, addr, value):
        self._retry(lambda: self.packet.write1ByteTxRx(self.port, dxl_id, addr, int(value)), f"write ID {dxl_id}")

    def _write2(self, dxl_id, addr, value):
        self._retry(lambda: self.packet.write2ByteTxRx(self.port, dxl_id, addr, int(value)), f"write ID {dxl_id}")

    # --- joint interface ---
    def read(self):
        """Present joint angles [deg]."""
        q = []
        for k, dxl_id in enumerate(self.MOTOR_IDS):
            ticks, _, _ = self._retry(
                lambda: self.packet.read2ByteTxRx(self.port, dxl_id, ADDR_PRESENT_POSITION), f"read ID {dxl_id}")
            q.append(self.SIGN[k] * (ticks * DEG_PER_TICK - self.OFFSET_DEG[k]))
        return q

    def move_j(self, q, wait=True, tol=2.0, timeout=10.0):
        """Move to joint angles q = [q1, q2, q3, q4] in degrees."""
        for k, (lo, hi) in enumerate(self.LIMITS):
            if not lo <= q[k] <= hi:
                raise ValueError(f"q{k+1} = {q[k]:.1f} deg is outside {lo}..{hi}")
        for k, dxl_id in enumerate(self.MOTOR_IDS):
            servo_deg = self.SIGN[k] * q[k] + self.OFFSET_DEG[k]
            self._write2(dxl_id, ADDR_GOAL_POSITION, round(servo_deg / DEG_PER_TICK))
        if not wait:
            return
        start = time.time()
        while time.time() - start < timeout:
            now = self.read()
            if all(abs(a - b) < tol for a, b in zip(now, q)):
                return now
            time.sleep(0.05)
        print("Warning: move timed out, now at", [round(a, 1) for a in now])
        return now

    def close(self, limp=False):
        if limp:
            for i in self.MOTOR_IDS:
                self.packet.write1ByteTxRx(self.port, i, ADDR_TORQUE_ENABLE, 0)
        self.port.closePort()

"""First IK test: send the stylus tip to a few points and check them with the Measure app.

Measure from the centre of the base:
  x = forward (away from the camera side), y = to the left, z = height above the table.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # find robot/

from robot.arm import Arm
from robot.kinematics import ik

TARGETS = [
    # (x,   y,   z,   w)       w = 0: stylus horizontal, w = -1: stylus pointing down
    (150,   0, 120,  0),       # the circle centre from Problem 3
    (150,   0,  60, -1),       # stylus pointing down, 6 cm above the table
    (140,  50,  60, -1),       # 5 cm to the left
    (140, -50,  60, -1),       # 5 cm to the right
]

arm = Arm(speed=0.1)
try:
    for x, y, z, w in TARGETS:
        q = ik(x, y, z, w)
        input(f"\nEnter -> tip to ({x}, {y}, {z}) mm, w={w}   q = {q.round(1)}")
        arm.move_j(q)
        print("  Measure the tip position now.")
    input("\nEnter -> back home")
    arm.move_j([0, 0, 0, 0])
finally:
    arm.close()

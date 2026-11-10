"""Move one joint at a time to +20 deg so you can compare with Figure 1.

Expected (DH convention from the report):
  q1 +20: whole arm turns counterclockwise, seen from above
  q2 +20: arm leans towards the camera side (-x0)
  q3 +20: upper part bends towards the camera side
  q4 +20: stylus tips towards the camera side
Positive q2..q4 lean AWAY from where the arm will type (+x0).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # find robot/

from robot.arm import Arm

TEST_DEG = 20
arm = Arm(speed=0.1)
try:
    arm.move_j([0, 0, 0, 0])
    for k in range(4):
        q = [0, 0, 0, 0]
        q[k] = TEST_DEG
        input(f"\nPress Enter to move joint {k+1} to +{TEST_DEG} deg...")
        arm.move_j(q)
        print("  now at:", [round(a, 1) for a in arm.read()])
        input("  Note the direction, then press Enter to go back to 0...")
        arm.move_j([0, 0, 0, 0])
    print("\nDone. If a joint went the wrong way, set its SIGN to -1 in arm.py.")
finally:
    arm.close()

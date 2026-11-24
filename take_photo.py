import time
import cv2
from robot.arm import Arm
from type_hello_test import camera_joint_angles, CAM_RHO, CAM_PHI, CAM_Z, CAMERA_INDEX

arm = Arm(speed=0.1)
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
try:
    arm.move_j(camera_joint_angles(CAM_RHO, CAM_PHI, CAM_Z))
    time.sleep(1.0)
    for _ in range(5):
        cap.read()
    ok, frame = cap.read()
    cv2.imwrite("tests/keyboard.png", frame)
    print("Saved tests/keyboard.png")
finally:
    cap.release()
    arm.move_j([0, 0, 0, 0])
    arm.close()
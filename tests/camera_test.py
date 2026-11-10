"""Move the camera above the keyboard (looking down) and show a live preview
with the recognised keys drawn on it (red dot = where the robot would press).

Press  s  to save the view with the keys drawn on it,  q  to quit (the arm then goes back home).
If you see the wrong camera (e.g. the MacBook's own), change CAMERA_INDEX to 1 or 2.
"""
import os, sys, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # find robot/

import cv2
from robot.arm import Arm
from robot.kinematics import ik, fk
from robot.vision import recognize_keys, draw

CAMERA_INDEX = 0
RECOGNITION_INTERVAL = 0.35

# Where the camera should be [mm]: x forward, y left, z above the top of the platform
CAM_X, CAM_Y, CAM_Z = 140, 0, 150

# With the stylus pointing down, the camera is 45 mm further out than the tip
# and 15 mm above it (frame {5} from Problem 1).
rho = math.hypot(CAM_X, CAM_Y)
scale = (rho - 45) / rho
q = ik(CAM_X * scale, CAM_Y * scale, CAM_Z - 15, w=-1)
print("Camera pose, joint angles:", q.round(1))

arm = Arm(speed=0.1)
cap = None
try:
    arm.move_j(q)
    print("Camera at", fk(q)[1][:3, 3].round(1), "mm, looking down")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {CAMERA_INDEX}. Try another index, "
                         "or allow camera access for Terminal / VS Code in System Settings.")

    print("Press s to save the recognised view, q to quit.")
    last_recognition = 0.0
    keys, info = None, None
    status = "Recognising keyboard..."
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No frame from the camera.")
            break

        # Recognition runs a few times per second; the video itself stays live
        now = time.time()
        if now - last_recognition >= RECOGNITION_INTERVAL:
            try:
                keys, info = recognize_keys(frame)
                missing = [("SPACE" if c == " " else c) for c in "HELOWRD " if c not in keys]
                status = f"{info['layout']} | {info['inliers']} keys fitted | "
                status += f"missing: {' '.join(missing)}" if missing else "all hello-world keys found"
            except (RuntimeError, cv2.error):
                keys, info = None, None
                status = "No keyboard found"
            last_recognition = now

        annotated = draw(frame, keys, info) if keys else frame.copy()
        color = (0, 200, 0) if "all hello" in status else (0, 0, 255)
        cv2.putText(annotated, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, color, 2, cv2.LINE_AA)
        cv2.imshow("keyboard recognition (s = save, q = quit)", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            name = f"keyboard_{time.strftime('%H%M%S')}_keys.png"
            cv2.imwrite(name, annotated)
            print("Saved", name, f"({annotated.shape[1]}x{annotated.shape[0]})")
        elif key == ord("q"):
            break
finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    arm.move_j([0, 0, 0, 0])
    arm.close()

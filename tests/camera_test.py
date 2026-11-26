"""Move the camera to the photo pose of the main program and show a live preview
with the letters the OCR read reliably (red dots). The status line runs the same
keyboard fit as type_hello.py: not every letter has to be read, the rest of the
keys (also ones outside the image) are computed from the fitted keyboard grid.

Uses the same pose (robot/setup.json "home") and resolution (640 x 480) as type_hello.py,
so what you see here is what the main program sees.
Press  s  to save the view,  q  to quit (the arm stays in the photo pose).
If you see the wrong camera (e.g. the MacBook's own), change CAMERA_INDEX to 1 or 2.
"""
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # find robot/

import cv2
import numpy as np
from robot.arm import Arm
from robot.keyboard import locate
from robot.vision import recognize_keys, require_tesseract

CAMERA_INDEX = 0
RECOGNITION_INTERVAL = 0.6          # s between recognitions; the video itself stays live

setup_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "robot", "setup.json")
with open(setup_path) as f:
    setup = json.load(f)
photo_pose = np.array(setup["home"])

require_tesseract()
arm = Arm()
cap = None
try:
    print("Photo pose, joint angles:", photo_pose.round(1))
    arm.move_j(photo_pose)
    joints = arm.angles()           # actual encoder angles, used for the pixel -> mm conversion

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {CAMERA_INDEX}. Try another index, "
                         "or allow camera access for Terminal / VS Code in System Settings.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press s to save the view, q to quit.")
    last, keys, status, good, size = 0.0, {}, "Recognising keyboard...", False, ""
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No frame from the camera.")
            break

        now = time.time()
        if now - last >= RECOGNITION_INTERVAL:
            h, w = frame.shape[:2]
            size = "" if (w, h) == (640, 480) else f" | camera gives {w}x{h}, main program needs 640x480!"
            try:
                keys = recognize_keys(frame)
            except RuntimeError:
                keys = {}
            try:
                # Same fit as the main program: every key position in mm from the robot base
                xy, fitted, rms = locate(frame, joints, setup)
                status = (f"{len(keys)} letters read | keyboard located: {fitted} fitted, "
                          f"residual {rms:.1f} mm | H at ({xy['H'][0]:.0f}, {xy['H'][1]:.0f}) mm")
                good = True
            except (ValueError, RuntimeError) as error:
                status, good = f"{len(keys)} letters read | NOT located: {error}", False
            status += size
            last = now

        view = frame.copy()
        for ch, (x, y) in keys.items():
            cv2.circle(view, (int(x), int(y)), 4, (0, 0, 255), -1)
            cv2.putText(view, ch, (int(x) + 6, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        color = (0, 200, 0) if good and not size else (0, 0, 255)
        cv2.putText(view, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        cv2.imshow("keyboard recognition (s = save, q = quit)", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            name = f"keyboard_{time.strftime('%H%M%S')}.png"
            cv2.imwrite(name, frame)
            print("Saved", name)
        elif key == ord("q"):
            break
finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    arm.close()

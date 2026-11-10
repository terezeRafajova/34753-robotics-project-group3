"""Problem 11a: recognise the keyboard with the camera and type "hello world".

What it does now (works without the stylus):
  1. Moves the camera above the keyboard, looking down, and recognises the keys.
  2. Calibrates the camera with the robot itself: moves the camera 20 mm outwards
     and 20 mm sideways and looks how the keys shift in the image. This gives the
     direction of the image axes relative to the robot (no ruler needed).
  3. Converts every key from pixels to robot coordinates [mm] and prints them.
Pressing the keys comes next, once the stylus is mounted (PRESS_KEYS below).

Run from the Project folder:   python type_hello.py
"""
import math
import time

import cv2
import numpy as np

from robot.arm import Arm
from robot.kinematics import ik, fk
from robot.vision import recognize_keys, draw

TEXT = "hello world"
CAMERA_INDEX = 0

# Camera pose for looking at the keyboard: distance from the base axis, angle, height [mm, deg]
CAM_RHO, CAM_PHI, CAM_Z = 140.0, 0.0, 150.0
CALIB_STEP = 20.0           # mm the camera moves during calibration

CAM_TO_TIP = 45.0           # with the stylus pointing down the camera is 45 mm further out than the tip
CAM_ABOVE_TIP = 15.0        # ... and 15 mm higher (frame {5}, Problem 1)

PRESS_KEYS = False          # set to True once the stylus is mounted and the key height is known


# ----------------------------------------------------------------------------- arm helpers
def camera_joint_angles(rho, phi_deg, z):
    """Joint angles that put the camera at (rho, phi, z) looking straight down."""
    phi = math.radians(phi_deg)
    tip_rho = rho - CAM_TO_TIP
    return ik(tip_rho * math.cos(phi), tip_rho * math.sin(phi), z - CAM_ABOVE_TIP, w=-1)


def grab_keys(cap, tries=10):
    """Let the arm settle, then return (frame, keys, info) from a fresh camera frame."""
    time.sleep(0.8)
    for _ in range(5):          # throw away old frames waiting in the camera buffer
        cap.read()
    for _ in range(tries):
        ok, frame = cap.read()
        if not ok:
            continue
        try:
            keys, info = recognize_keys(frame)
            return frame, keys, info
        except RuntimeError:
            pass
    raise RuntimeError("Could not recognise the keyboard from this camera pose.")


# ----------------------------------------------------------------------------- calibration
def mean_shift(keys_a, keys_b):
    """Average pixel shift of the keys seen in both views."""
    common = [c for c in keys_a if c in keys_b and c != " "]
    if len(common) < 5:
        raise RuntimeError("Too few keys visible in both calibration views.")
    return np.mean([np.subtract(keys_b[c], keys_a[c]) for c in common], axis=0)


def build_pixel_matrix(shift_radial, shift_tangential, mm_per_px):
    """2x2 matrix J: offset [mm] in the camera's (radial, tangential) axes -> pixels.

    Moving the camera by +d makes the keys move by -J d in the image.
    The direction of the radial axis comes from the radial move; the tangential
    axis is perpendicular to it, and the tangential move only decides the sign
    (the image may be mirrored). The scale comes from the 19 mm key pitch.
    """
    s = 1.0 / mm_per_px                                   # pixels per mm
    col_r = -shift_radial / np.linalg.norm(shift_radial) * s
    perp = np.array([-col_r[1], col_r[0]])
    col_t = perp if np.dot(-shift_tangential, perp) > 0 else -perp
    return np.column_stack([col_r, col_t])


def keys_to_robot(keys, J, image_size, cam_xy, phi_deg):
    """Pixel positions -> robot (x, y) [mm], assuming the image centre is straight below the camera."""
    w, h = image_size
    centre = np.array([w / 2, h / 2])
    J_inv = np.linalg.inv(J)
    phi = math.radians(phi_deg)
    r_hat = np.array([math.cos(phi), math.sin(phi)])
    t_hat = np.array([-math.sin(phi), math.cos(phi)])
    out = {}
    for ch, px in keys.items():
        d_r, d_t = J_inv @ (np.asarray(px) - centre)
        out[ch] = np.asarray(cam_xy) + d_r * r_hat + d_t * t_hat
    return out


# ----------------------------------------------------------------------------- main
def main():
    arm = Arm(speed=0.1)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    try:
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera {CAMERA_INDEX}.")

        # 1. Look at the keyboard
        q = camera_joint_angles(CAM_RHO, CAM_PHI, CAM_Z)
        arm.move_j(q)
        frame, keys, info = grab_keys(cap)
        print(f"Keyboard: {info['layout']}, {info['inliers']} keys fitted, "
              f"{info['mm_per_px']:.3f} mm/pixel")

        # 2. Calibrate: radial move (base angle fixed), then a sideways move (base turns)
        arm.move_j(camera_joint_angles(CAM_RHO + CALIB_STEP, CAM_PHI, CAM_Z))
        _, keys_r, _ = grab_keys(cap)
        d_phi = math.degrees(CALIB_STEP / CAM_RHO)        # sideways arc of CALIB_STEP mm
        arm.move_j(camera_joint_angles(CAM_RHO, CAM_PHI + d_phi, CAM_Z))
        _, keys_t, _ = grab_keys(cap)
        arm.move_j(q)                                     # back to the viewing pose

        J = build_pixel_matrix(mean_shift(keys, keys_r), mean_shift(keys, keys_t), info["mm_per_px"])
        print("Pixel matrix J (mm in camera radial/tangential -> pixels):\n", J.round(2))

        # 3. Keys in robot coordinates
        cam_xy = fk(q)[1][:2, 3]                          # camera position from the kinematics
        h, w = frame.shape[:2]
        key_xy = keys_to_robot(keys, J, (w, h), cam_xy, CAM_PHI)

        print("\nKeys for", repr(TEXT), "in robot coordinates [mm]:")
        for ch in dict.fromkeys(TEXT.upper()):
            name = "SPACE" if ch == " " else ch
            if ch in key_xy:
                x, y = key_xy[ch]
                print(f"  {name:5s}  x = {x:6.1f}   y = {y:6.1f}   (distance from base {math.hypot(x, y):5.1f})")
            else:
                print(f"  {name:5s}  not in view!")

        view = draw(frame, keys, info)
        for ch in TEXT.upper():
            if ch in keys:
                x, y = key_xy[ch]
                px, py = keys[ch]
                cv2.putText(view, f"({x:.0f},{y:.0f})", (int(px) - 35, int(py) + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite("type_hello_view.png", view)
        print("\nSaved type_hello_view.png")

        if PRESS_KEYS:
            raise SystemExit("Pressing is not implemented yet: mount the stylus first.")

    finally:
        cap.release()
        arm.move_j([0, 0, 0, 0])
        arm.close()


if __name__ == "__main__":
    main()

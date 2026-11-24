"""Problem 11a: recognise the keyboard with the camera and type "hello world".

What it does now (works without the stylus):
  1. Moves the camera above the keyboard, looking down, and recognises the keys.
  2. Calibrates the camera with the robot itself: moves the camera 20 mm outwards
     and 20 mm sideways and looks how the keys shift in the image. This gives the
     direction of the image axes relative to the robot (no ruler needed).
  3. Converts every key from pixels to robot coordinates [mm] and prints them.
  4. (--type) Presses the keys with the stylus.

Run from the Project folder:
    python type_hello.py                 look, calibrate, print key positions (no pressing)
    python type_hello.py --find-height   jog the stylus down onto the G key to measure KEY_Z
    python type_hello.py --type          type "hello world" (needs KEY_Z)
"""
import argparse
import math
import time

import cv2
import numpy as np

from robot.arm import Arm
from robot.kinematics import ik, fk
from robot.vision import recognize_keys, draw

TEXT = "hello world"
CAMERA_INDEX = 1

# Camera pose for looking at the keyboard: distance from the base axis, angle, height [mm, deg]
CAM_RHO, CAM_PHI, CAM_Z = 140.0, 0.0, 150.0
CALIB_STEP = 20.0           # mm the camera moves during calibration

CAM_TO_TIP = 45.0           # with the stylus pointing down the camera is 45 mm further out than the tip
CAM_ABOVE_TIP = 15.0        # ... and 15 mm higher (frame {5}, Problem 1)

KEY_Z = 54             # z of the key tops in robot coordinates [mm]; measure with --find-height
KEY_OFFSET = (-20.0,-15.0)     # (x, y) correction [mm] added to every key; also measured with --find-height
HOVER = 25.0                # mm above the key tops while moving between keys
PRESS_DEPTH = 3.0           # aim this far below the key top (key travel is ~3-4 mm)
PRESS_TIME = 0.4            # s to hold a key down
STEP_MM = 10.0              # straight-line moves are split into steps of at most this


# ----------------------------------------------------------------------------- arm helpers
def camera_joint_angles(rho, phi_deg, z):
    """Joint angles that put the camera at (rho, phi, z) looking straight down."""
    phi = math.radians(phi_deg)
    tip_rho = rho - CAM_TO_TIP
    return ik(tip_rho * math.cos(phi), tip_rho * math.sin(phi), z - CAM_ABOVE_TIP, w=-1)


def grab_keys(cap, views=3, tries=15):
    """Let the arm settle, then return (frame, keys, info) from fresh camera frames.

    The keys are recognised in several frames and the median position is used, so one
    poor layout fit (a few misread letters) doesn't spoil the result.
    """
    time.sleep(0.8)
    for _ in range(5):          # throw away old frames waiting in the camera buffer
        cap.read()
    results = []
    for _ in range(tries):
        ok, frame = cap.read()
        if not ok:
            continue
        try:
            keys, info = recognize_keys(frame)
        except RuntimeError:
            continue
        results.append((frame, keys, info))
        if len(results) == views:
            break
    if not results:
        raise RuntimeError("Could not recognise the keyboard from this camera pose.")
    frame, _, info = max(results, key=lambda r: r[2]["inliers"])
    common = set.intersection(*(set(k) for _, k, _ in results))
    keys = {c: tuple(np.median([k[c] for _, k, _ in results], axis=0)) for c in common}
    return frame, keys, info


# ----------------------------------------------------------------------------- calibration
def mean_shift(keys_a, keys_b, image_size):
    """Average pixel shift of the keys seen inside the image in both views."""
    w, h = image_size
    inside = lambda p: 0 <= p[0] < w and 0 <= p[1] < h
    common = [c for c in keys_a if c in keys_b and c != " " and inside(keys_a[c]) and inside(keys_b[c])]
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


# ----------------------------------------------------------------------------- pressing keys
def tip_q(x, y, z):
    """Joint angles for the stylus tip at (x, y, z), pointing straight down."""
    return ik(x, y, z, w=-1)


def tip_position(arm):
    """Where the stylus tip actually is now (from the measured joint angles)."""
    return fk(arm.read())[0][:3, 3]


def move_tip_line(arm, p_from, p_to):
    """Move the tip along a straight line in small steps.

    One big move_j would let each servo run on its own, and the tip can dip on the
    way (and hit the keyboard). Short steps keep it close to the straight line.
    """
    p_from, p_to = np.asarray(p_from, float), np.asarray(p_to, float)
    n = max(1, int(math.ceil(np.linalg.norm(p_to - p_from) / STEP_MM)))
    for k in range(1, n + 1):
        # in-between points: carry on once roughly there (smooth); last point: arrive properly
        arm.move_j(tip_q(*(p_from + (p_to - p_from) * k / n)), tol=3.0 if k == n else 8.0)
    return p_to


def check_reachable(key_xy, text, key_z):
    """Before moving: every key in the text is visible and reachable within the joint limits."""
    missing = [c for c in dict.fromkeys(text.upper()) if c not in key_xy]
    if missing:
        raise RuntimeError(f"Keys not found in the image: {missing}")
    for ch in dict.fromkeys(text.upper()):
        x, y = key_xy[ch]
        for z in (key_z + HOVER, key_z + 2.0, key_z - PRESS_DEPTH):
            q = tip_q(x, y, z)                         # raises ValueError if out of reach
            for k, (lo, hi) in enumerate(Arm.LIMITS):
                if not lo <= q[k] <= hi:
                    raise RuntimeError(f"Key {ch!r} at z={z:.0f}: q{k+1} = {q[k]:.1f} deg is outside {lo}..{hi}")


def press(arm, x, y, key_z):
    """From hovering above a key: go down, press, come back up."""
    move_tip_line(arm, (x, y, key_z + HOVER), (x, y, key_z + 2.0))     # just above the key top
    arm.move_j(tip_q(x, y, key_z - PRESS_DEPTH), wait=False)           # the key stops the tip, so don't wait
    time.sleep(PRESS_TIME)
    move_tip_line(arm, (x, y, key_z - PRESS_DEPTH), (x, y, key_z + HOVER))


def type_text(arm, key_xy, text, key_z):
    check_reachable(key_xy, text, key_z)
    pos = tip_position(arm)
    for ch in text.upper():
        x, y = key_xy[ch]
        above = (x, y, key_z + HOVER)
        if pos[2] < key_z + HOVER - 1:                 # never travel sideways below hover height
            pos = move_tip_line(arm, pos, (pos[0], pos[1], key_z + HOVER))
        pos = move_tip_line(arm, pos, above)
        press(arm, x, y, key_z)
        print(f"  pressed {'SPACE' if ch == ' ' else ch}")


def find_key_height(arm, key_xy, z_start=80.0, key="G"):
    """Jog the stylus onto the centre of a key; print KEY_Z and KEY_OFFSET."""
    x0, y0 = key_xy[key]
    x, y, z = x0, y0, z_start
    move_tip_line(arm, tip_position(arm), (x, y, z))
    print(f"""
The stylus should be above {key}. Move it until it is on the CENTRE of {key}
and JUST touches the key top (without pressing it). Commands, then Enter:
  (just Enter)  1 mm down          d 10 / u 3   10 mm down / 3 mm up
  o 5 / i 5     5 mm out (away from the robot) / in (toward the robot)
  l 5 / r 5     5 mm left / right (standing behind the robot, looking at the keyboard)
  q             done
Lift the stylus (u 5) before moving it sideways.""")
    while True:
        cmd = input(f"  x = {x:6.1f}  y = {y:6.1f}  z = {z:6.1f} mm > ").strip().lower()
        if cmd == "q":
            break
        parts = cmd.split()
        try:
            amount = float(parts[1]) if len(parts) > 1 else 1.0
        except ValueError:
            continue
        what = parts[0][0] if parts else "d"
        step = {"d": (0, 0, -1), "u": (0, 0, 1), "o": (1, 0, 0), "i": (-1, 0, 0),
                "l": (0, 1, 0), "r": (0, -1, 0)}.get(what)
        if step is None:
            print("   Unknown command. Use d, u, o, i, l, r followed by a number, e.g.  l 5")
            continue
        nx, ny, nz = x + amount * step[0], y + amount * step[1], z + amount * step[2]
        try:
            arm.move_j(tip_q(nx, ny, nz), wait=False)  # don't wait: touching the key may stop the servo
        except ValueError as e:
            print(f"   Can't go there ({e}). Try a smaller step.")
            continue
        x, y, z = nx, ny, nz
        time.sleep(0.4)
    off = (KEY_OFFSET[0] + x - x0, KEY_OFFSET[1] + y - y0)
    print("\nPut these at the top of the script:")
    print(f"KEY_Z = {z:.1f}")
    print(f"KEY_OFFSET = ({off[0]:.1f}, {off[1]:.1f})")
    move_tip_line(arm, (x, y, z), (x, y, z + HOVER))


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--find-height", action="store_true", help="measure KEY_Z by jogging onto the G key")
    parser.add_argument("--type", action="store_true", help=f"type {TEXT!r}")
    args = parser.parse_args()
    if args.type and KEY_Z is None:
        raise SystemExit("Set KEY_Z first (run with --find-height).")

    arm = Arm(speed=0.1)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    try:
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera {CAMERA_INDEX}.")

        # 1. Look at the keyboard
        q = camera_joint_angles(CAM_RHO, CAM_PHI, CAM_Z)
        arm.move_j(q)
        frame, keys, info = grab_keys(cap)
        print(f"Keyboard: {info['layout']}, {info['inliers']} keys fitted, "
              f"{info['mm_per_px']:.3f} mm/pixel")

        # 2. Calibrate: radial moves (base angle fixed) inwards and outwards, then a sideways move
        h, w = frame.shape[:2]
        arm.move_j(camera_joint_angles(CAM_RHO - CALIB_STEP, CAM_PHI, CAM_Z))
        _, keys_in, _ = grab_keys(cap)
        arm.move_j(camera_joint_angles(CAM_RHO + CALIB_STEP, CAM_PHI, CAM_Z))
        _, keys_out, _ = grab_keys(cap)
        d_phi = math.degrees(CALIB_STEP / CAM_RHO)        # sideways arc of CALIB_STEP mm
        arm.move_j(camera_joint_angles(CAM_RHO, CAM_PHI + d_phi, CAM_Z))
        _, keys_t, _ = grab_keys(cap)
        arm.move_j(q)                                     # back to the viewing pose

        shift_r = mean_shift(keys_in, keys_out, (w, h))   # camera moved 2 * CALIB_STEP outwards
        J = build_pixel_matrix(shift_r, mean_shift(keys, keys_t, (w, h)), info["mm_per_px"])
        angle = math.degrees(math.atan2(J[1, 0], J[0, 0]))
        moved = np.linalg.norm(shift_r) * info["mm_per_px"]
        print(f"Calibration: image rotation {angle:.1f} deg, radial move seen as {moved:.1f} mm "
              f"(commanded {2 * CALIB_STEP:.0f} mm)")

        # 3. Keys in robot coordinates
        cam_xy = fk(arm.read())[1][:2, 3]                 # camera position from the MEASURED joint angles
        key_xy = keys_to_robot(keys, J, (w, h), cam_xy, CAM_PHI)
        key_xy = {c: p + np.asarray(KEY_OFFSET) for c, p in key_xy.items()}   # measured correction

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

        # 4. Use the stylus
        if args.find_height:
            find_key_height(arm, key_xy)
        elif args.type:
            print(f"\nTyping {TEXT!r} (click into a text editor on the keyboard's computer first!)")
            type_text(arm, key_xy, TEXT, KEY_Z)
            move_tip_line(arm, tip_position(arm), tip_position(arm) + [0, 0, 40])   # lift clear

    finally:
        cap.release()
        arm.move_j([0, 0, 0, 0])
        arm.close()


if __name__ == "__main__":
    main()
"""Calibrate robot/setup.json for the keyboard in front of the robot (no touching).

    python tests/calibrate_camera.py                 standard keyboard (19.05 mm keys)
    python tests/calibrate_camera.py --pitch 18.4    other key spacing [mm]
    python tests/calibrate_camera.py --height -45    with the measured height of the key tops

The arm takes one photo from the photo pose and reads the letters.
- Without --height it finds the height at which the keys are exactly `pitch` mm apart.
- With --height (z of the key tops in mm; z = 0 is the top of the red base, below it is
  negative) it prints the key spacing the camera sees. About 19 mm means the camera
  calibration fits this robot; something very different means it does not.
Then it asks whether to overwrite robot/setup.json.

Only the keyboard (height, key spacing, key travel) is changed. The camera calibration
(photo pose and projection) is reused as it is.

    python tests/calibrate_camera.py --aim
Corrects the key positions for this robot's camera. The arm hovers 15 mm above each
letter of "hello world"; you nudge the tip with the keyboard until it is above the centre
of the key (k skips a key). From the nudges it computes a rotation + shift + stretch, shown
before it asks to save it in setup.json (xy_correction). Running it again refines the saved
correction. Torque stays on, nothing is pressed.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                    # find robot/ and type_hello.py
from robot.kinematics import fk
from robot.vision import layout_units, recognize_keys, require_tesseract


def fit_scale(seen, joints, setup, z):
    """Key spacing seen in the photo divided by the expected one, if the keys are at height z.
    Same calculation as robot/keyboard.py: 1.0 means height and spacing agree."""
    camera = fk(joints)[1]
    rays = camera[:3, :3].T @ np.column_stack(([1, 0, 0], [0, 1, 0], np.array([0, 0, z]) - camera[:3, 3]))
    base_to_image = np.asarray(setup["views"][0]["projection"]) @ rays
    units = layout_units()
    names = sorted(seen)
    board = np.array([np.subtract(units[k], units["H"]) * [setup["column_pitch"], -setup["row_pitch"]]
                      for k in names])
    pixels = np.array([[seen[k] for k in names]], dtype=float)
    base = cv2.perspectiveTransform(pixels, np.linalg.inv(base_to_image))[0]
    fit, _ = cv2.estimateAffinePartial2D(board, base, ransacReprojThreshold=2.5)
    if fit is None:
        raise RuntimeError("The letters do not fit a keyboard grid.")
    return float(np.linalg.norm(fit[:, 0]))


def solve_height(seen, joints, setup):
    """Bisection on the key height until the keys appear at the given spacing."""
    lo, hi = setup["surface_z"] - 100, setup["surface_z"] + 60     # lower keys look smaller
    s_lo, s_hi = fit_scale(seen, joints, setup, lo), fit_scale(seen, joints, setup, hi)
    if not (s_lo - 1) * (s_hi - 1) < 0:
        raise RuntimeError(f"No height between {lo:.0f} and {hi:.0f} mm fits. Check the key spacing (--pitch).")
    for _ in range(40):
        mid = (lo + hi) / 2
        s_mid = fit_scale(seen, joints, setup, mid)
        if (s_mid - 1) * (s_lo - 1) > 0:
            lo, s_lo = mid, s_mid
        else:
            hi = mid
    return (lo + hi) / 2


AIM_KEYS = ("W", "E", "R", "D", "H", "L", "O")   # the keys of "hello world"
AIM_HELP = """
Nudge the tip until it is straight above the CENTRE of the key (look from above):
  w / s  towards the number row / towards the space bar
  a / d  towards Q (left) / towards P (right)
  1 / 3 / 5  step in mm (start: 3)
  Enter  centred -> next key        k  skip this key
"""


def fit_correction(src, dst):
    """2x3 correction that maps src points onto dst points (least squares).
    3+ keys: affine (rotation, shift, and stretch: fixes wrong key spacing or height too).
    2 keys: rotation + shift.  1 key: shift only."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if len(src) >= 3:
        m, *_ = np.linalg.lstsq(np.column_stack([src, np.ones(len(src))]), dst, rcond=None)
        return m.T
    return fit_rigid(src, dst)


def fit_rigid(src, dst):
    """Rotation + shift (2x3) that maps src points onto dst points (least squares)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if len(src) == 1:
        return np.hstack([np.eye(2), (dst[0] - src[0])[:, None]])
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    return np.hstack([r, (cd - r @ cs)[:, None]])


def aim(args, setup, setup_file):
    import time
    from robot.arm import Arm
    from robot.keyboard import locate
    from robot.kinematics import ik
    from robot.motion import Motion
    from type_hello import photograph

    arm = Arm(args.port)
    motion = Motion(arm, setup, 30)
    try:
        motion.home()
        # Same as type_hello.py: if one photo can't be read, try the other camera poses
        for i, pose in enumerate([setup["home"]] + [v["joints"] for v in setup["views"]]):
            if i:
                motion.move(np.asarray(pose))
            try:
                frame, joints = photograph(args.camera, arm)
                xy, count, rms = locate(frame, joints, setup)
                break
            except (ValueError, RuntimeError) as e:
                print(f"Photo {i + 1}: {e}")
        else:
            raise SystemExit("No photo could be read. Turn the keyboard backlight off, keep hands "
                             "out of the picture, check with tests/camera_test.py.")
        print(f"{count} letters fitted, residual {rms:.2f} mm")
        right = xy["P"] - xy["Q"]
        right /= np.linalg.norm(right)
        up = np.array([-right[1], right[0]])
        if np.dot(up, xy["Q"] - xy["A"]) < 0:
            up = -up
        z = setup["surface_z"] + 15                      # hover 15 mm above the key tops
        print(AIM_HELP)

        def go(p, first=False):
            q = ik(p[0], p[1], z, pitch_deg=-180)
            if first:
                try:
                    motion.move(q)                       # big move: with clearance checks
                except RuntimeError as e:                # stopped a few mm short: fine here,
                    print("  (arm stopped a few mm short; nudge from where it is)")   # you nudge anyway
            else:
                arm.validate(q)
                arm.command(q, [30] * 4)                         # small nudge
                time.sleep(0.6)

        src, dst, names = [], [], []
        for key in AIM_KEYS:
            p = np.array(xy[key], float)
            go(p, first=True)
            p = fk(arm.angles())[0][:2, 3]               # continue from where it really is
            step = 3.0
            while True:
                cmd = input(f"  {key} > ").strip().lower()
                if cmd == "":
                    # Where the tip REALLY is (encoders), not where it was told to go: small
                    # nudges are sometimes not executed, or the joint stops a bit short.
                    real = fk(arm.angles())[0][:2, 3]
                    src.append(xy[key]); dst.append(real); names.append(key)
                    shift = real - xy[key]
                    print(f"     {key}: moved {np.linalg.norm(shift):.1f} mm from the computed position")
                    break
                if cmd == "k":
                    break
                if cmd in ("1", "3", "5"):
                    step = float(cmd)
                    continue
                d = {"w": up, "s": -up, "a": -right, "d": right}.get(cmd)
                if d is None:
                    print("  unknown command")
                    continue
                try:
                    go(p + step * d)
                    p = p + step * d
                    real = fk(arm.angles())[0][:2, 3]
                    if np.linalg.norm(real - p) > 2:     # did not follow: let the next nudge
                        print(f"  (arm is {np.linalg.norm(real - p):.1f} mm behind; "
                              "look at the tip, not the count)")
                except ValueError as e:
                    print("  can't go there:", e)
            motion.move(ik(p[0], p[1], z + 15, pitch_deg=-180), "lift")

        if not src:
            print("No keys centred; nothing changed.")
            return
        m = fit_correction(src, dst)
        angle = np.degrees(np.arctan2(m[1, 0], m[0, 0]))
        stretch = np.linalg.svd(m[:, :2], compute_uv=False)
        left = [np.linalg.norm(m[:, :2] @ a + m[:, 2] - b) for a, b in zip(src, dst)]
        print("After the correction, each key is off by [mm]: "
              + "  ".join(f"{k} {e:.1f}" for k, e in zip(names, left)))
        print(f"\nCorrection: rotation {angle:+.1f} deg, shift ({m[0, 2]:+.1f}, {m[1, 2]:+.1f}) mm, "
              f"stretch {stretch[0]:.3f} / {stretch[1]:.3f}"
              + ("  (only 1 key: shift only)" if len(src) == 1 else "")
              + ("  (2 keys: no stretch)" if len(src) == 2 else ""))
        old = np.vstack([np.asarray(setup.get("xy_correction", np.hstack([np.eye(2), np.zeros((2, 1))]))),
                         [0, 0, 1]])
        total = (np.vstack([m, [0, 0, 1]]) @ old)[:2]   # on top of any earlier correction
        if input("Save this correction in robot/setup.json? [y/N] ").strip().lower() == "y":
            setup["xy_correction"] = total.round(5).tolist()
            setup_file.write_text(json.dumps(setup, indent=2))
            print("Saved. Next: python type_hello.py run")
        else:
            print("Not changed.")
    finally:
        try:
            motion.home()
        finally:
            arm.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pitch", type=float, default=19.05, help="key spacing in mm (standard: 19.05)")
    parser.add_argument("--height", type=float, help="measured z of the key tops in mm (negative)")
    parser.add_argument("--travel", type=float, default=3.5, help="key travel in mm (membrane: about 3.5)")
    parser.add_argument("--camera", type=int, default=1 if sys.platform == "win32" else 0)
    parser.add_argument("--port", default=None, help="default: first /dev/tty.usbmodem*")
    parser.add_argument("--aim", action="store_true", help="correct the key positions for this robot")
    parser.add_argument("--image", type=Path, help="offline test with a saved photo (no robot)")
    parser.add_argument("--joints", type=float, nargs=4, help="encoder angles for --image")
    args = parser.parse_args()

    require_tesseract()
    setup_file = ROOT / "robot" / "setup.json"
    setup = json.loads(setup_file.read_text())
    if args.aim:
        return aim(args, setup, setup_file)
    new = dict(setup, column_pitch=args.pitch, row_pitch=args.pitch, travel=args.travel)

    # 1. One photo from the photo pose
    if args.image:
        frame = cv2.imread(str(args.image))
        joints = np.array(args.joints or setup["views"][0]["joints"])
    else:
        from robot.arm import Arm
        from type_hello import photograph
        arm = Arm(args.port)
        try:
            arm.move_j(np.asarray(setup["home"]))
            frame, joints = photograph(args.camera, arm)
        finally:
            arm.close()
        cv2.imwrite("calibration-photo.png", frame)
    seen = recognize_keys(frame)
    print(f"{len(seen)} letters read: {' '.join(sorted(seen))}")

    # 2. Keyboard height and key spacing
    if args.height is not None:
        z = args.height
        pitch = args.pitch * fit_scale(seen, joints, new, z)
        print(f"With the keys at z = {z:.1f} mm the camera sees a key spacing of {pitch:.2f} mm.")
        if not 17.5 <= pitch <= 20:
            print("That is not a normal key spacing: the camera calibration does not fit this "
                  "robot or camera. setup.json was not changed.")
            return
        new.update(column_pitch=pitch, row_pitch=pitch)
    else:
        z = solve_height(seen, joints, new)
        print(f"Key spacing {args.pitch} mm -> key tops at z = {z:.1f} mm "
              f"(previous setup: {setup['surface_z']:.1f} mm)")
    new.update(surface_z=z, surface_samples={"H": z})

    # 3. Overwrite setup.json?
    if args.image:
        return
    print(f"\nNew values: key tops z = {z:.1f} mm, key spacing {new['column_pitch']:.2f} mm, "
          f"travel {new['travel']:.1f} mm")
    if input("Overwrite robot/setup.json with these values? [y/N] ").strip().lower() != "y":
        print("Not changed.")
        return
    setup_file.write_text(json.dumps(new, indent=2))
    print("Saved robot/setup.json. Next: python type_hello.py plan")


if __name__ == "__main__":
    main()

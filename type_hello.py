"""DTU 34753, problem 11a: recognise the keyboard and type "hello world".

Run from the Project folder, in a terminal:
    python type_hello.py plan     # photo + plan, moves the arm but presses nothing
    python type_hello.py run      # photo + plan + type
    python type_hello.py home     # just go home
"""
import argparse
import glob
import json
import os
from pathlib import Path
import sys
import time
import math

import cv2
import numpy as np

from robot.vision import require_tesseract
from robot.arm import Arm
from robot.keyboard import locate
from robot.motion import Motion, check_segment, key_poses

TEXT = "hello world"
SETUP = Path(__file__).parent / "robot" / "setup.json"


def default_port():
    """The USB2AX shows up as /dev/tty.usbmodem* on a Mac."""
    ports = sorted(glob.glob("/dev/tty.usbmodem*"))
    return ports[0] if ports else None


def make_input_reader():
    """Non-blocking read of the characters typed into this terminal (the physical keystrokes)."""
    # Switch the terminal to character mode (no Enter, no echo) and restore it on exit.
    import atexit
    import select
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    atexit.register(termios.tcsetattr, fd, termios.TCSADRAIN, old)

    def read_input():
        chars = ""
        while select.select([fd], [], [], 0)[0]:
            chars += os.read(fd, 64).decode(errors="replace")
        return chars
    return read_input


def read_setup():
    return json.loads(SETUP.read_text())


def photograph(camera, arm):
    cap = cv2.VideoCapture(camera)
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {camera}.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(5):
            cap.read()
        for _ in range(6):
            before = arm.angles()
            ok, frame = cap.read()
            after = arm.angles()
            if ok and frame is not None and np.std(frame) > 8 and max(abs(after-before)) <= 1:
                cv2.imwrite("last-camera.png", frame)
                return frame, (before+after)/2
        raise ValueError(f"Camera gave dark, empty or moving frames. Check camera {camera} and the USB connection.")
    finally:
        cap.release()


def plan(frame, joints, setup):
    xy, count, rms = locate(frame, joints, setup)
    print(f"{count} letters fitted, residual {rms:.2f} mm")
    poses = {}
    for ch in dict.fromkeys(TEXT):
        point = xy[ch.upper()]
        print(f"{ch if ch != ' ' else 'SPACE':5} XY {point.round(1)} mm")
        # Heights were touched once on this keyboard. Only XY comes from the live frame.
        samples = setup["surface_samples"]
        distances = np.array([np.linalg.norm(point - xy[key]) for key in samples])
        weights = 1 / np.maximum(distances, 1e-3)**2
        top = float(np.average(list(samples.values()), weights=weights))
        poses[ch] = key_poses(point, top, setup)
    # Check the complete phrase before the first press, including E to L and return.
    home = np.asarray(setup["home"])
    current = home
    for ch in TEXT:
        hover, bottom = poses[ch]
        check_segment(current, hover, "travel", setup)
        check_segment(hover, bottom, "press", setup)
        check_segment(bottom, hover, "lift", setup)
        current = hover
    check_segment(current, home, "travel", setup)
    return poses


def type_text(motion, poses, read_input):
    typed = ""
    for ch in TEXT:
        hover, bottom = poses[ch]
        motion.move(hover)
        extra = read_input()
        if extra:
            raise RuntimeError(f"Unexpected input before pressing {ch!r}: {extra!r}")
        events = []
        def key_event():
            events.extend(read_input())
            return bool(events)

        try:
            motion.move(bottom, "press", key_event)
        except RuntimeError:
            # A missing event or stalled press must still release the key.
            motion.arm.hold()
            motion.move(hover, "lift")
            raise
        motion.move(hover, "lift")
        events.extend(read_input())
        got = "".join(events)
        if got != ch:
            raise RuntimeError(f"After {typed!r}, expected {ch!r}, received {got!r}.")
        typed += got
        print(f"Verified physical input: {typed!r}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "run", "home"])
    parser.add_argument("--port", default=default_port(), help="default: first /dev/tty.usbmodem*")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--speed", type=float, default=60, help="transfer speed in joint degrees/s")
    args = parser.parse_args()
    setup = read_setup()
    if not math.isfinite(args.speed) or not 0 < args.speed <= 1023 * 0.666:
        parser.error("--speed must be in the AX joint-mode speed command range (0 to 681 degrees/s, excluding zero).")
    if args.mode == "run":
        if not sys.stdin.isatty():
            raise RuntimeError("Start in a terminal: the typed characters are the feedback for each press.")
        read_input = make_input_reader()
        if read_input():
            raise RuntimeError("Start in an empty, focused terminal.")
    if args.mode != "home":
        require_tesseract()
    arm = Arm(args.port)
    motion = Motion(arm, setup, args.speed)
    try:
        motion.home()
        print(f"Home reached. Encoders (deg): {arm.angles().round(2)}", flush=True)
        if args.mode == "home":
            return
        for index, pose in enumerate([setup["home"]] + [v["joints"] for v in setup["views"]]):
            if index:
                motion.move(np.asarray(pose))
            try:
                frame, joints = photograph(args.camera, arm)
                print(f"Exposure {index+1}, encoders (deg): {joints.round(2)}", flush=True)
                poses = plan(frame, joints, setup)
                break
            except (ValueError, RuntimeError) as exc:
                print(f"Photo {index + 1}: {exc}", flush=True)
        else:
            motion.home()
            raise ValueError("No calibrated view produced a usable plan.")
        motion.home()
        if args.mode == "run":
            start = time.monotonic()
            type_text(motion, poses, read_input)
            motion.home()
            print(f"Typed {TEXT!r} and returned home in {time.monotonic()-start:.1f} s.")
    except BaseException:
        try:
            arm.hold()
        except (OSError, ValueError) as hold_error:
            print(f"Could not confirm position hold: {hold_error}", file=sys.stderr)
        raise
    finally:
        arm.port.closePort()


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, KeyboardInterrupt) as error:
        sys.exit(f"Stopped: {error or 'interrupted'}")

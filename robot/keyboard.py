"""Map the new photograph to key centres, in millimetres from the robot base."""
import cv2
import numpy as np

from .kinematics import fk
from .vision import layout_units, recognize_keys


def locate(frame, joints, setup):
    if list(frame.shape[1::-1]) != setup["resolution"]:
        raise ValueError(f"Camera resolution must be {setup['resolution']}, got {frame.shape[1::-1]}.")
    view = min(setup["views"], key=lambda v: np.linalg.norm(np.subtract(v["joints"], joints)))
    if np.max(np.abs(np.subtract(view["joints"], joints))) > 3:
        raise ValueError("The photo was taken outside the calibrated camera poses.")
    camera = fk(joints)[1]
    origin = np.array([0, 0, setup["surface_z"]])
    rays = camera[:3, :3].T @ np.column_stack(([1, 0, 0], [0, 1, 0], origin - camera[:3, 3]))
    base_to_image = np.asarray(view["projection"]) @ rays
    observed = recognize_keys(frame)
    names = sorted(observed)
    units = layout_units()
    # Keyboard right and keyboard front form a left-handed pair in base XY.
    board = {key: np.subtract(uv, units["H"]) * [setup["column_pitch"], -setup["row_pitch"]]
             for key, uv in units.items()}
    if len(names) < 7:
        raise ValueError(f"Only {len(names)} unambiguous letters were read.")
    metric = np.array([board[k] for k in names])
    pixels = np.array([[observed[k] for k in names]], dtype=float)
    base = cv2.perspectiveTransform(pixels, np.linalg.inv(base_to_image))[0]
    fit, mask = cv2.estimateAffinePartial2D(metric, base, ransacReprojThreshold=2.5)
    if fit is None or mask is None:
        raise ValueError("Letter centres do not fit the keyboard.")
    good = mask.ravel().astype(bool)
    a, b = metric[good], base[good]
    if (len(a) < max(7, 0.75 * len(names)) or np.ptp(a[:, 1]) < setup["row_pitch"] * 0.95
            or np.ptp(a[:, 0]) < 3 * setup["column_pitch"]):
        raise ValueError("The photo needs matching letters across multiple rows and columns.")
    if abs(np.linalg.norm(fit[:, 0]) - 1) > 0.07:
        raise ValueError("Measured key spacing disagrees with the setup. Check keyboard height and camera.")
    # Remove scale from the final fit: the keyboard can only translate and rotate.
    u, _, vt = np.linalg.svd((a - a.mean(0)).T @ (b - b.mean(0)))
    rotation = u @ vt
    if np.linalg.det(rotation) <= 0:
        raise ValueError("The calibrated camera and live image disagree on orientation.")
    offset = b.mean(0) - a.mean(0) @ rotation
    xy = {key: point @ rotation + offset for key, point in board.items()}
    residual = np.sqrt(np.mean(np.sum((a @ rotation + offset - b)**2, axis=1)))
    # Correction measured on this robot (tests/calibrate_camera.py --aim): rotation + shift
    # in base XY that fixes the difference between this camera and the calibrated one.
    if "xy_correction" in setup:
        m = np.asarray(setup["xy_correction"])
        xy = {key: m[:, :2] @ p + m[:, 2] for key, p in xy.items()}
    return xy, len(a), residual

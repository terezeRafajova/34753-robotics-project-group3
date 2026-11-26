"""Read unambiguous legends on the dark lab keyboard with OpenCV and Tesseract.

I/L and Y/Z are unnecessary for locating the shared QWERTY/QWERTZ letter grid.
The final metric fit uses measured glyph centres in the original camera frame.
"""
import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import pytesseract

def layout_units():
    """Letter positions measured in column and row intervals from Q."""
    units = {}
    for row, letters in enumerate(("QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM")):
        for col, key in enumerate(letters):
            units[key] = (col + (0, 0.26, 0.75)[row], row)
    units[" "] = (4.75, 3.0)  # a point inside the wide bar below B
    return units


def require_tesseract():
    """Resolve the OCR executable from an override, PATH or an install directory."""
    command = os.environ.get("TESSERACT_CMD") or pytesseract.pytesseract.tesseract_cmd
    executable = shutil.which(command)
    if executable is None and sys.platform == "win32" and command == "tesseract":
        folders = [Path(os.environ[name]) for name in ("ProgramFiles", "ProgramFiles(x86)")
                   if os.environ.get(name)]
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            folders.append(Path(local_app_data) / "Programs")
        for folder in folders:
            candidate = folder / "Tesseract-OCR" / "tesseract.exe"
            if candidate.is_file():
                executable = str(candidate)
                break
    if executable is None:
        raise FileNotFoundError(
            f"Tesseract OCR executable not found ({command}). "
            "Install Tesseract OCR or set TESSERACT_CMD to the full path to its executable."
        )
    pytesseract.pytesseract.tesseract_cmd = executable


def find_letter_blobs(gray, threshold=150, scale=1.0):
    """Bright connected blobs about the size of a printed letter."""
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 9 * scale <= h <= 46 * scale and 4 * scale <= w <= 46 * scale and area >= 3 * h:
            glyph = 255 - (labels[y:y+h, x:x+w] == i).astype(np.uint8) * 255  # black letter on white
            blobs.append({"cx": x + w / 2, "cy": y + h / 2, "w": w, "h": h, "glyph": glyph})
    return blobs


def read_letters(blobs):
    """Read each glyph on its own line in a batched Tesseract request."""
    if not blobs:
        return
    cell = max(3 * int(np.median([b["h"] for b in blobs])),
               max(max(b["w"], b["h"]) for b in blobs) + 8)
    # Keep the enlarged sheet below Tesseract's image dimension limit.
    batch_size = max(1, 12000 // cell)
    cfg = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for start in range(0, len(blobs), batch_size):
        batch = blobs[start:start + batch_size]
        sheet = np.full((cell * len(batch), cell), 255, np.uint8)
        for k, b in enumerate(batch):
            g = b["glyph"]
            oy, ox = k * cell + (cell - g.shape[0]) // 2, (cell - g.shape[1]) // 2
            sheet[oy:oy + g.shape[0], ox:ox + g.shape[1]] = g
        enlarged = cv2.resize(sheet, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        sheet = cv2.GaussianBlur(enlarged, (3, 3), 0)
        d = pytesseract.image_to_data(sheet, config=cfg, output_type=pytesseract.Output.DICT)
        for k, text in enumerate(d["text"]):
            text = text.strip()
            idx = int((d["top"][k] + d["height"][k] / 2) / (2 * cell))
            if len(text) == 1 and 0 <= idx < len(batch):
                batch[idx]["ch"] = text


def fit_layout(blobs):
    """Use the known row arrangement to discard conflicting OCR readings."""
    grid = layout_units()
    letters = [b for b in blobs if b.get("ch") in grid and b["ch"] not in "ILYZ "]
    if len({b["ch"] for b in letters}) < 7:
        return None
    src = np.float32([grid[b["ch"]] for b in letters])
    dst = np.float32([[b["cx"], b["cy"]] for b in letters])
    threshold = 0.6 * np.median([b["h"] for b in letters])
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, threshold)
    if homography is None or not np.isfinite(homography).all():
        return None
    corners = np.float32([[[0, 0], [9, 0], [9, 3], [0, 3]]])
    if np.any(corners[0] @ homography[2, :2] + homography[2, 2] <= 0):
        return None
    tl, tr, br, bl = cv2.perspectiveTransform(corners, homography)[0]
    right = np.array([tr - tl, br - bl])
    down = np.array([bl - tl, br - tr])
    if (np.any(right[:, 0] <= 0.5 * np.abs(right[:, 1])) or
            np.any(down[:, 1] <= 0.5 * np.abs(down[:, 0]))):
        return None
    projected = cv2.perspectiveTransform(src[None], homography)[0]
    errors = np.linalg.norm(projected - dst, axis=1)
    best = {}
    for k in np.argsort(errors):
        if mask[k, 0]:
            best.setdefault(letters[k]["ch"], k)
    if len(best) < 7 or len({grid[k][1] for k in best}) < 2:
        return None
    indices = list(best.values())
    return (len(best), -float(np.median(errors[indices]))), {ch: dst[k] for ch, k in best.items()}


def keyboard_angle(gray):
    """Estimate the keyboard axes modulo 90 degrees from long edges."""
    length = max(20, min(gray.shape) // 8)
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30,
                            minLineLength=length, maxLineGap=length // 5)
    if lines is None:
        return 0.0
    segments = lines.reshape(-1, 4)
    deltas = segments[:, 2:] - segments[:, :2]
    lengths = np.linalg.norm(deltas, axis=1)
    if not np.any(lengths > 0):
        return 0.0
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    direction = np.average(np.exp(4j * angles), weights=lengths)
    return float(np.degrees(np.angle(direction)) / 4) if abs(direction) >= 0.45 else 0.0


def orient_frame(gray, angle, mirrored):
    """Turn the image without cropping and return its pixel transform."""
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1)
    if mirrored:
        matrix = matrix @ np.array([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]])
    corners = np.float32([[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]])
    oriented = cv2.transform(corners, matrix)[0]
    matrix[:, 2] -= oriented.min(axis=0)
    size = tuple(np.ceil(np.ptp(oriented, axis=0)).astype(int) + 1)
    return cv2.warpAffine(gray, matrix, size), matrix


def try_orientations(gray, offsets, scale, threshold=150):
    """OCR each candidate and rank it using agreement with the physical layout."""
    best = None
    for mirrored in (False, True):
        for offset in offsets:
            for turn in (0, 90, 180, 270):
                angle = turn - offset if mirrored else turn + offset
                view, matrix = orient_frame(gray, angle, mirrored)
                blobs = find_letter_blobs(view, threshold=threshold, scale=scale)
                if len(blobs) < 6:
                    continue
                read_letters(blobs)
                fit = fit_layout(blobs)
                if fit and (best is None or fit[0] > best[0]):
                    score, centres = fit
                    inverse = cv2.invertAffineTransform(matrix)
                    best = score, {ch: inverse @ [*pixel, 1] for ch, pixel in centres.items()}
                    if score[0] >= 12 and score[1] > -3:
                        return best
    return best


def recognize_keys(image):
    """Return OCR inliers as a mapping of letter to original image pixel."""
    require_tesseract()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = min(gray.shape) / 480
    angle = keyboard_angle(gray)
    offsets = (0, angle) if abs(angle) >= 3 else (0,)
    result = try_orientations(gray, offsets, scale)
    if result is None or result[0][0] < 9:
        # Remove broad glare from the keycaps to isolate the smaller bright legends.
        width = max(3, 2 * round(15 * scale) + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
        local_contrast = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        variants = ((local_contrast, 65), (equalized, 150))
        for view, threshold in variants:
            retry = try_orientations(view, offsets, scale, threshold)
            if retry is not None and (result is None or retry[0] > result[0]):
                result = retry
                if result[0][0] >= 9:
                    break
        if result is None:
            for view, threshold in variants:
                result = try_orientations(view, (-30, -15, 15, 30, 45), scale, threshold)
                if result is not None:
                    break
    if result is None:
        raise RuntimeError("Could not recognise the keyboard: too few letters fit a layout.")

    return result[1]

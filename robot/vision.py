"""Find the letter keys of a keyboard in a camera image.

1. Find bright letter-shaped blobs (the white letters on the dark keys).
2. Read them with OCR (Tesseract), all in one call.
3. Fit a keyboard layout to the letters that were read, so a few misread
   or unread keys don't matter. QWERTY and QWERTZ are both tried; the one
   that fits best wins.

Result: pixel position of every letter key + the space bar.
Run it on a photo:   python robot/vision.py tests/keyboard_141042.png
"""
import sys
import time

import cv2
import numpy as np
import pytesseract

KEY_PITCH_MM = 19.0     # centre-to-centre distance of neighbouring keys

# Key positions in "key units" (1 unit = 19 mm). Row 0 = Q row, row 1 = A row, row 2 = bottom row.
ROW_OFFSET = [0.0, 0.26, 0.75]
LAYOUTS = {
    "QWERTZ": ["QWERTZUIOP", "ASDFGHJKL", "YXCVBNM"],
    "QWERTY": ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"],
}
SPACE_UNITS = (4.75, 3.0)   # centre of the space bar, under B


def layout_units(rows):
    units = {}
    for r, letters in enumerate(rows):
        for c, ch in enumerate(letters):
            units[ch] = (c + ROW_OFFSET[r], float(r))
    units[" "] = SPACE_UNITS
    return units


def find_letter_blobs(gray, threshold=150):
    """Bright connected blobs about the size of a printed letter."""
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 18 <= h <= 32 and 4 <= w <= 32 and area >= 60:
            glyph = 255 - (labels[y:y+h, x:x+w] == i).astype(np.uint8) * 255  # black letter on white
            blobs.append({"cx": x + w / 2, "cy": y + h / 2, "w": w, "h": h, "glyph": glyph})
    return blobs


def read_letters(blobs):
    """OCR all blobs in a single Tesseract call: each glyph on its own line."""
    if not blobs:
        return
    cell = 3 * int(np.median([b["h"] for b in blobs]))
    sheet = np.full((cell * len(blobs), cell), 255, np.uint8)
    for k, b in enumerate(blobs):
        g = b["glyph"]
        oy, ox = k * cell + (cell - g.shape[0]) // 2, (cell - g.shape[1]) // 2
        sheet[oy:oy + g.shape[0], ox:ox + g.shape[1]] = g
    sheet = cv2.GaussianBlur(cv2.resize(sheet, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST), (3, 3), 0)
    cfg = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    d = pytesseract.image_to_data(sheet, config=cfg, output_type=pytesseract.Output.DICT)
    for k, text in enumerate(d["text"]):
        text = text.strip()
        idx = int((d["top"][k] + d["height"][k] / 2) / (2 * cell))
        if len(text) == 1 and 0 <= idx < len(blobs):
            blobs[idx]["ch"] = text


def fit_layout(blobs):
    """Fit each layout to the read letters with RANSAC; keep the best one."""
    best = None
    for name, rows in LAYOUTS.items():
        units = layout_units(rows)
        pairs = [(units[b["ch"]], (b["cx"], b["cy"])) for b in blobs if b.get("ch") in units]
        if len(pairs) < 6:
            continue
        src = np.float32([p[0] for p in pairs])
        dst = np.float32([p[1] for p in pairs])
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 15.0)
        if H is None:
            continue
        inliers = int(mask.sum())
        if best is None or inliers > best["inliers"]:
            best = {"layout": name, "H": H, "inliers": inliers, "read": len(pairs), "units": units}
    return best


def recognize_keys(image):
    """Return (keys, info): keys maps each letter and ' ' to its pixel (x, y)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blobs = find_letter_blobs(gray)
    read_letters(blobs)
    fit = fit_layout(blobs)
    if fit is None:
        raise RuntimeError("Could not recognise the keyboard: too few letters were read.")
    names = list(fit["units"])
    pts = cv2.perspectiveTransform(np.float32([fit["units"][c] for c in names]).reshape(-1, 1, 2), fit["H"])
    h, w = gray.shape
    keys = {c: tuple(p[0]) for c, p in zip(names, pts) if 0 <= p[0][0] < w and 0 <= p[0][1] < h}
    # Pixel size of one key (19 mm), measured in the middle of the keyboard
    a, b = cv2.perspectiveTransform(np.float32([[[4, 1]], [[5, 1]]]), fit["H"])
    px_per_key = float(np.linalg.norm(a[0] - b[0]))
    info = {"layout": fit["layout"], "letters_read": fit["read"], "inliers": fit["inliers"],
            "mm_per_px": KEY_PITCH_MM / px_per_key, "blobs": blobs}
    return keys, info


def draw(image, keys, info):
    out = image.copy()
    for b in info["blobs"]:
        if "ch" in b:   # what OCR read (yellow)
            cv2.putText(out, b["ch"], (int(b["cx"]) - 30, int(b["cy"]) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    for ch, (x, y) in keys.items():   # final key centres (red dot + green label)
        cv2.circle(out, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(out, "SPACE" if ch == " " else ch, (int(x) + 9, int(y) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "keyboard.png"
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Could not read {path}")
    t0 = time.time()
    keys, info = recognize_keys(img)
    print(f"Done in {time.time() - t0:.2f} s")
    print(f"Layout: {info['layout']}  (OCR read {info['letters_read']} letters, "
          f"{info['inliers']} fit the layout)")
    print(f"Scale:  {info['mm_per_px']:.3f} mm per pixel")
    for ch in dict.fromkeys("HELLO WORLD"):
        if ch in keys:
            print(f"  {'SPACE' if ch == ' ' else ch}: pixel ({keys[ch][0]:.0f}, {keys[ch][1]:.0f})")
        else:
            print(f"  {ch}: not in view!")
    out_path = path.rsplit(".", 1)[0] + "_keys.png"
    cv2.imwrite(out_path, draw(img, keys, info))
    print("Saved", out_path)

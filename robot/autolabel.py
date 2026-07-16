#!/usr/bin/env python3
"""Auto-label snack-rotator frames with color segmentation -> YOLO boxes.

Classes: 0=black 1=red 2=blue 3=silver 4=x
Writes YOLO .txt next to labels/, plus overlay .jpg previews for review.
"""
import cv2, numpy as np, sys, glob, os

FRAMES = os.path.expanduser("~/snack-rotator/dataset/frames")
LABELS = os.path.expanduser("~/snack-rotator/dataset/labels_auto")
PREVIEW = os.path.expanduser("~/snack-rotator/dataset/preview")
os.makedirs(LABELS, exist_ok=True)
os.makedirs(PREVIEW, exist_ok=True)

NAMES = ["black", "red", "blue", "silver", "x"]
COLORS = [(40, 40, 40), (0, 0, 255), (255, 80, 0), (180, 180, 180), (0, 255, 0)]

def biggest_blob(mask, min_area):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best, area = None, min_area
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > area:
            area = stats[i, cv2.CC_STAT_AREA]
            x, y, w, h = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
            best = (x, y, w, h)
    return best

def process(path, save_preview):
    img = cv2.imread(path)
    H, W = img.shape[:2]
    small = cv2.resize(img, (W // 2, H // 2))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    sh, sw = small.shape[:2]

    # --- disc: tan/brown cardboard ---
    tan = ((h > 5) & (h < 30) & (s > 40) & (s < 160) & (v > 90)).astype(np.uint8) * 255
    tan = cv2.morphologyEx(tan, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(tan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    disc = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(disc) < 0.05 * sh * sw:
        return None
    hull = cv2.convexHull(disc)
    hull_fill = np.zeros((sh, sw), np.uint8)
    cv2.fillPoly(hull_fill, [hull], 255)
    disc_mask = cv2.dilate(hull_fill, np.ones((21, 21), np.uint8))   # loose: colored tapes may overhang
    strict_mask = cv2.erode(hull_fill, np.ones((13, 13), np.uint8))  # strict: black/silver look like sheet/bg
    (cx, cy), r = cv2.minEnclosingCircle(hull)

    boxes = []  # (cls, x, y, w, h) in small coords
    min_area = 0.0015 * sh * sw

    # --- tapes inside disc ---
    red = ((h > 3) & (h < 17) & (s > 85) & (v > 90)).astype(np.uint8) * 255
    blue = ((h > 95) & (h < 135) & (s > 30) & (v > 140)).astype(np.uint8) * 255
    black = ((s < 40) & (v > 45) & (v < 165)).astype(np.uint8) * 255
    silver = ((s < 28) & (v > 165)).astype(np.uint8) * 255
    for cls, mask, region in ((1, red, disc_mask), (2, blue, disc_mask),
                              (0, black, strict_mask), (3, silver, strict_mask)):
        m = cv2.bitwise_and(mask, region)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        b = biggest_blob(m, min_area)
        if b:
            boxes.append((cls,) + b)

    # --- X: dark strokes on white sheet, outside the disc ---
    white = ((s < 40) & (v > 170)).astype(np.uint8) * 255
    white = cv2.bitwise_and(white, cv2.bitwise_not(disc_mask))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    if n > 1 and stats[1:, cv2.CC_STAT_AREA].max() > 0.02 * sh * sw:
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        sheet = (lab == i).astype(np.uint8) * 255
        # fill the X-stroke holes so they count as "inside the sheet"
        sheet = cv2.morphologyEx(sheet, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
        sheet = cv2.erode(sheet, np.ones((9, 9), np.uint8))
        dark = (v < 110).astype(np.uint8) * 255
        dark = cv2.bitwise_and(dark, sheet)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
        xb = biggest_blob(dark, min_area * 0.6)
        if xb:
            # X should be roughly square-ish strokes, not a huge blob
            _, _, bw, bh = xb
            if 0.3 < bw / max(bh, 1) < 3.5 and bw < 0.5 * sw:
                boxes.append((4,) + xb)

    # write YOLO label (normalized, full-res frame)
    name = os.path.splitext(os.path.basename(path))[0]
    with open(os.path.join(LABELS, name + ".txt"), "w") as f:
        for cls, x, y, w, hh in boxes:
            f.write(f"{cls} {(x + w / 2) / sw:.6f} {(y + hh / 2) / sh:.6f} {w / sw:.6f} {hh / sh:.6f}\n")

    if save_preview:
        vis = small.copy()
        cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 255), 2)
        for cls, x, y, w, hh in boxes:
            cv2.rectangle(vis, (x, y), (x + w, y + hh), COLORS[cls], 2)
            cv2.putText(vis, NAMES[cls], (x, y - 6), 0, 0.7, COLORS[cls], 2)
        cv2.imwrite(os.path.join(PREVIEW, name + ".jpg"), vis)
    return len(boxes)

files = sorted(glob.glob(FRAMES + "/*.jpg"))
counts = {}
for i, p in enumerate(files):
    n = process(p, save_preview=(i % 12 == 0))
    counts[n] = counts.get(n, 0) + 1
print("frames:", len(files), "boxes-per-frame histogram:", dict(sorted(counts.items(), key=lambda kv: (kv[0] is None, kv[0]))))

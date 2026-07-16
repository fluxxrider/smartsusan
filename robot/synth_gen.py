#!/usr/bin/env python3
"""Generate synthetic snack-scene training data by pasting real snack crops
onto real rig frames. Real X in each frame is auto-labeled with marker_v4.

Output: dataset_synth/{train,val}/{images,labels} + data.yaml
Classes: 0=bar 1=bear 2=juice 3=welchs 4=x
"""
import glob, math, os, random, shutil
import cv2
import numpy as np
from ultralytics import YOLO

random.seed(7)
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

CLS = {"bar": 0, "bear": 1, "juice": 2, "welchs": 3, "x": 4}
CROPS = {
    "bar": [cv2.imread("crops/good/bar.jpg")],
    "bear": [cv2.imread("crops/good/bear.jpg")],
    "juice": [cv2.imread("crops/good/juice_a.jpg"), cv2.imread("crops/good/juice_b.jpg")],
    "welchs": [cv2.imread("crops/good/welchs.jpg")],
}
mx = YOLO("runs/marker_v4/best.pt")

def paste(bg, crop, cx, cy, angle, scale, gain):
    ch, cw = crop.shape[:2]
    M = cv2.getRotationMatrix2D((cw / 2, ch / 2), angle, scale)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(ch * sin + cw * cos), int(ch * cos + cw * sin)
    M[0, 2] += nw / 2 - cw / 2
    M[1, 2] += nh / 2 - ch / 2
    rot = cv2.warpAffine(crop, M, (nw, nh), borderValue=(0, 255, 0))
    mask = ~((rot[:, :, 0] < 10) & (rot[:, :, 1] > 245) & (rot[:, :, 2] < 10))
    mask = cv2.erode(mask.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    rot = np.clip(rot.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    H, W = bg.shape[:2]
    x1, y1 = int(cx - nw / 2), int(cy - nh / 2)
    x2, y2 = x1 + nw, y1 + nh
    if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
        return None
    roi = bg[y1:y2, x1:x2]
    m3 = cv2.merge([mask] * 3) > 0
    roi[m3] = rot[m3]
    return (x1, y1, x2, y2)

def disc_geometry(dets):
    pts = [(v[0], v[1]) for n, v in dets.items() if n in ("black", "red", "blue", "silver")]
    if len(pts) < 3:
        return None
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    r = max(math.dist((cx, cy), p) for p in pts) * 1.15
    return cx, cy, r

frames = sorted(glob.glob("dataset2/raw_frames/*.jpg"))
random.shuffle(frames)
out_n = 0
shutil.rmtree("dataset_synth", ignore_errors=True)
for split in ("train", "val"):
    for sub in ("images", "labels"):
        os.makedirs(f"dataset_synth/{split}/{sub}", exist_ok=True)

for f in frames:
    r = mx.predict(f, device="cpu", conf=0.20, verbose=False)[0]
    dets = {}
    for b in r.boxes:
        n = mx.names[int(b.cls)]
        c = float(b.conf)
        if n not in dets or c > dets[n][2]:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            dets[n] = ((x1 + x2) / 2, (y1 + y2) / 2, c, (x1, y1, x2, y2))
    geo = disc_geometry(dets)
    if geo is None or "x" not in dets:
        continue
    cx, cy, rad = geo
    img = cv2.imread(f)
    H, W = img.shape[:2]

    labels = []
    # real X, labeled via marker_v4
    x1, y1, x2, y2 = dets["x"][3]
    labels.append((CLS["x"], (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))

    # paste the four snacks at random rim positions
    placed = []
    ok_all = True
    for name in ("bar", "bear", "juice", "welchs"):
        crop = random.choice(CROPS[name])
        for attempt in range(80):
            ang_pos = random.uniform(0, 2 * math.pi)
            rr = random.uniform(0.30, 0.80) * rad
            px, py = cx + rr * math.cos(ang_pos), cy + rr * math.sin(ang_pos)
            if any(math.dist((px, py), q) < 0.42 * rad for q in placed):
                continue
            box = paste(img, crop, px, py, random.uniform(0, 360),
                        random.uniform(0.6, 1.1), random.uniform(0.75, 1.2))
            if box:
                placed.append((px, py))
                bx1, by1, bx2, by2 = box
                labels.append((CLS[name], (bx1 + bx2) / 2 / W, (by1 + by2) / 2 / H,
                               (bx2 - bx1) / W, (by2 - by1) / H))
                break
        else:
            ok_all = False
            break
    if not ok_all:
        continue

    out_n += 1
    split = "val" if out_n % 8 == 0 else "train"
    base = f"synth_{out_n:04d}"
    cv2.imwrite(f"dataset_synth/{split}/images/{base}.jpg", img)
    with open(f"dataset_synth/{split}/labels/{base}.txt", "w") as o:
        for c, a, b, w, h in labels:
            o.write(f"{c} {a:.6f} {b:.6f} {w:.6f} {h:.6f}\n")

with open("dataset_synth/data.yaml", "w") as y:
    y.write(f"path: {os.path.abspath('dataset_synth')}\ntrain: train/images\nval: val/images\n"
            "names:\n  0: bar\n  1: bear\n  2: juice\n  3: welchs\n  4: x\n")
print(f"generated {out_n} synthetic frames")

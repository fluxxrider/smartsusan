#!/usr/bin/env python3
"""Two-round bootstrap training for the smartsusan marker detector.

Round 1: train on the small clean dataset (dataset/).
Round 2: pseudo-label ALL raw frames with the round-1 model, retrain on the
         expanded set. Output: runs/round2/weights/best.pt

    pip install ultralytics
    python bootstrap.py            # uses CUDA if available

Send back runs/round2/weights/best.pt
"""
import glob, os, shutil, statistics

import torch
from ultralytics import YOLO

DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")

# ---------- round 1: seed training on the clean split ----------
m1 = YOLO("yolov8n.pt")
m1.train(data=os.path.abspath("dataset/data.yaml"), epochs=40, imgsz=640,
         device=DEVICE, batch=16, project="runs", name="round1", plots=False)

# ---------- pseudo-label all raw frames with the round-1 model ----------
FLOORS = {"x": 0.30, "black": 0.40, "red": 0.50, "blue": 0.50, "silver": 0.40}
CLS = {"black": 0, "red": 1, "blue": 2, "silver": 3, "x": 4}
model = YOLO("runs/round1/weights/best.pt")

fs = sorted(glob.glob("raw_frames/*.jpg"))
labeled = {}
for f in fs:
    r = model.predict(f, device=DEVICE, conf=0.25, imgsz=640, verbose=False)[0]
    best = {}
    for b in r.boxes:
        n = model.names[int(b.cls)]
        c = float(b.conf)
        if c >= FLOORS[n] and (n not in best or c > best[n][0]):
            best[n] = (c, b.xywhn[0].tolist())
    if len(best) == 5:
        labeled[f] = best

areas = {n: [] for n in CLS}
for best in labeled.values():
    for n, (c, (cx, cy, w, h)) in best.items():
        areas[n].append(w * h)
med = {n: statistics.median(a) for n, a in areas.items()}
clean = {f: b for f, b in labeled.items()
         if all(0.4 * med[n] <= v[1][2] * v[1][3] <= 2.5 * med[n] for n, v in b.items())}
print(f"round 2 dataset: {len(clean)} of {len(fs)} frames")

for split in ("train", "val"):
    for sub in ("images", "labels"):
        os.makedirs(f"dataset_r2/{split}/{sub}", exist_ok=True)
names = sorted(clean)
val = set(names[::8])   # every 8th frame -> val
for f in names:
    split = "val" if f in val else "train"
    base = os.path.splitext(os.path.basename(f))[0]
    shutil.copy(f, f"dataset_r2/{split}/images/{base}.jpg")
    with open(f"dataset_r2/{split}/labels/{base}.txt", "w") as out:
        for n, (c, (cx, cy, w, h)) in clean[f].items():
            out.write(f"{CLS[n]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
with open("dataset_r2/data.yaml", "w") as y:
    y.write("path: .\ntrain: train/images\nval: val/images\n"
            "names:\n  0: black\n  1: red\n  2: blue\n  3: silver\n  4: x\n")

# ---------- round 2: the real training ----------
m2 = YOLO("yolov8n.pt")
m2.train(data=os.path.abspath("dataset_r2/data.yaml"), epochs=60, imgsz=640,
         device=DEVICE, batch=16, project="runs", name="round2", plots=False)
metrics = m2.val(device=DEVICE)
print("FINAL mAP50:", round(metrics.box.map50, 3))
for i, name in metrics.names.items():
    print("FINAL", name, "AP50:", round(metrics.box.ap50[i], 3))
print("\n>>> send back: runs/round2/weights/best.pt <<<")

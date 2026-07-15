#!/usr/bin/env python3
"""Capture training frames from the overhead Logitech while the platter spins.

    python3 capture_frames.py [minutes] [cam_index]     # defaults: 4 min, cam 0

Saves ~2 frames/sec to ~/snack-rotator/dataset2/raw_frames/
"""
import cv2, os, sys, time

minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
cam_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
out = os.path.expanduser("~/snack-rotator/dataset2/raw_frames")
os.makedirs(out, exist_ok=True)

cap = cv2.VideoCapture(cam_idx)
assert cap.isOpened(), f"camera {cam_idx} not available"
for _ in range(10):
    cap.read()

n = 0
deadline = time.time() + minutes * 60
print(f"capturing ~2 fps for {minutes} min - spin the platter, move the X, vary things!")
while time.time() < deadline:
    ok, f = cap.read()
    if ok:
        n += 1
        cv2.imwrite(f"{out}/logi_{n:04d}.jpg", f)
        if n % 20 == 0:
            print(f"  {n} frames, {int(deadline - time.time())}s left")
    time.sleep(0.5)
cap.release()
print(f"done: {n} frames in {out}")

#!/usr/bin/env python3
"""Probe camera indices 0-4: print resolution, save one snapshot each."""
import cv2, os

out = os.path.expanduser("~/snack-rotator/camprobe")
os.makedirs(out, exist_ok=True)
for i in range(5):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"cam {i}: not available")
        continue
    for _ in range(10):
        cap.read()
    ok, f = cap.read()
    if ok:
        print(f"cam {i}: {f.shape[1]}x{f.shape[0]}  -> camprobe/cam{i}.jpg")
        cv2.imwrite(f"{out}/cam{i}.jpg", f)
    else:
        print(f"cam {i}: opened but no frame")
    cap.release()

#!/usr/bin/env python3
"""Live framing viewer: watch the overhead camera while you aim it.

    python3 frame_view.py            # cam 0 (Logitech)
    python3 frame_view.py --cam 1

Keys:  d = toggle detection overlay (snacks + x)
       s = save snapshot to camprobe/cam_view.jpg
       q = quit
"""
import argparse, os, sys, time
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser()
p.add_argument("--cam", type=int, default=0)
args = p.parse_args()

cap = cv2.VideoCapture(args.cam)
assert cap.isOpened(), f"camera {args.cam} not available"

detect_on = True
models = None
last_dets, last_det_t = {}, 0.0
COL = {"bar": (0, 200, 255), "bear": (0, 140, 255), "juice": (255, 200, 0),
       "welchs": (180, 0, 255), "x": (0, 220, 0),
       "black": (80, 80, 80), "red": (0, 0, 230), "blue": (230, 130, 0), "silver": (200, 200, 200)}

print("framing view - d: detection, s: snapshot, q: quit")
while True:
    ok, f = cap.read()
    if not ok:
        continue
    H, W = f.shape[:2]

    if detect_on and time.time() - last_det_t > 0.7:
        if models is None:
            from deliver import load_model, detect
            models = load_model(snacks=True)
        r_dets = {}
        try:
            from deliver import detect as _detect
            raw = _detect(models, f)
            r_dets = raw
        except Exception:
            pass
        last_dets, last_det_t = r_dets, time.time()

    if detect_on:
        for n, (cx, cy, c) in last_dets.items():
            col = COL.get(n, (255, 255, 255))
            cv2.circle(f, (int(cx), int(cy)), 16, col, 3)
            cv2.putText(f, f"{n} {c:.2f}", (int(cx) + 18, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        need = ["bar", "bear", "juice", "welchs", "x"]
        missing = [n for n in need if n not in last_dets]
        msg = "ALL 5 VISIBLE - framing good!" if not missing else "missing: " + " ".join(missing)
        cv2.putText(f, msg, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 220, 0) if not missing else (0, 0, 255), 3)

    # framing guides: centre cross + safe margin box
    cv2.drawMarker(f, (W // 2, H // 2), (255, 255, 0), cv2.MARKER_CROSS, 40, 2)
    cv2.rectangle(f, (int(W * 0.06), int(H * 0.06)), (int(W * 0.94), int(H * 0.94)),
                  (255, 255, 0), 1)
    cv2.imshow("framing", f)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord("d"):
        detect_on = not detect_on
    if k == ord("s"):
        os.makedirs("camprobe", exist_ok=True)
        cv2.imwrite(os.path.expanduser("~/snack-rotator/camprobe/cam_view.jpg"), f)
        print("snapshot saved")
cap.release()
cv2.destroyAllWindows()

#!/usr/bin/env python3
"""Person-following arrow: webcam finds a person, servo points the arrow at them.

    python3 follow.py             # webcam 0, pretrained COCO person detector
    python3 follow.py --cam 1

Mount an arrow on the servo horn so that at 90 deg it points where the
camera looks. Keys in the window: f = flip direction, q = quit.
"""
import argparse, os, sys, time
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deliver import Board

FOV_DEG = 70          # approx horizontal field of view of the webcam
SMOOTH = 0.35         # EMA factor (higher = snappier)
DEADBAND = 3          # deg; don't chase jitter
MOVE_EVERY = 0.15     # s between servo commands

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--fov", type=float, default=FOV_DEG)
    args = p.parse_args()

    from ultralytics import YOLO
    model = YOLO(os.path.expanduser("~/snack-rotator/yolov8n.pt")
                 if os.path.exists(os.path.expanduser("~/snack-rotator/yolov8n.pt"))
                 else "yolov8n.pt")

    cam = cv2.VideoCapture(args.cam)
    assert cam.isOpened(), f"camera {args.cam} not available"
    board = Board()
    board.goto(90)

    sign = 1
    angle = 90.0
    last_move = 0.0
    print("following - arrow tracks the person. f=flip, q=quit")

    while True:
        ok, frame = cam.read()
        if not ok:
            continue
        H, W = frame.shape[:2]
        r = model.predict(frame, device="cpu", conf=0.4, imgsz=416,
                          classes=[0], verbose=False)[0]   # class 0 = person
        best = None
        for b in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            if best is None or area > best[4]:
                best = (x1, y1, x2, y2, area, float(b.conf))
        status = "no person"
        if best:
            x1, y1, x2, y2, _, conf = best
            cx = (x1 + x2) / 2
            off = (cx / W - 0.5) * args.fov * sign     # deg off camera centre
            target = max(0, min(180, 90 + off))
            angle += SMOOTH * (target - angle)
            if abs(angle - board_angle_cache(board)) > DEADBAND and time.time() - last_move > MOVE_EVERY:
                board.goto(angle)
                last_move = time.time()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 3)
            cv2.putText(frame, f"person {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)
            status = f"arrow -> {angle:.0f} deg (off-centre {off:+.1f})"
            # arrow HUD
            cv2.arrowedLine(frame, (W // 2, H - 40), (int(cx), H - 40), (0, 220, 0), 4, tipLength=0.06)
        cv2.putText(frame, status + "   [f]lip [q]uit", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow("arrow follower", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("f"):
            sign = -sign
            print("direction flipped")
    board.cmd("REL")
    cam.release()
    cv2.destroyAllWindows()

_cache = {"a": 90, "t": 0.0}
def board_angle_cache(board):
    # avoid a serial round trip every frame; refresh occasionally
    if time.time() - _cache["t"] > 1.0:
        _cache["a"] = board.angle()
        _cache["t"] = time.time()
    return _cache["a"]

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Dual-camera snack delivery: overhead cam tracks the disc, webcam tracks the PERSON.
The requested tape is rotated toward wherever the person is standing.

Setup: iPhone overhead (markers + X sheet visible), MacBook webcam facing the
person area, X sheet placed in line with where the webcam looks (it anchors the
two reference frames; keep it there).

    python3 duo.py                     # overhead cam 1, person cam 0
    python3 duo.py --over 2 --person 0

Window keys:  k/r/b/s = deliver black/red/blue/silver to the person
              f = flip webcam direction    q = quit
"""
import argparse, os, sys, time
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deliver import load_model, detect, scene_angles, needed_rotation, make_board, TAPES, TOL_DEG

FOV_DEG = 70
KEYS = {ord("k"): "black", ord("r"): "red", ord("b"): "blue", ord("s"): "silver"}
BOX = {"black": (60, 60, 60), "red": (0, 0, 230), "blue": (230, 130, 0),
       "silver": (200, 200, 200), "x": (0, 220, 0)}

def open_cam(i):
    c = cv2.VideoCapture(i)
    assert c.isOpened(), f"camera {i} not available"
    # 720p: two simultaneous streams at 1080p overwhelm Continuity Camera
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    for _ in range(10):  # warm up; first frames are often torn
        c.read()
    return c

def good_frame(f):
    """Reject torn/grey frames (Continuity glitch): grey fill has ~zero variance."""
    import numpy as np
    bottom = f[f.shape[0] // 2:]
    return bottom.std() > 12

def read_valid(cam, tries=4):
    for _ in range(tries):
        ok, f = cam.read()
        if ok and good_frame(f):
            return f
    return None

def person_offset(person_model, frame, fov):
    """Person's bearing offset (deg) from webcam centre, or None."""
    r = person_model.predict(frame, device="cpu", conf=0.4, imgsz=416,
                             classes=[0], verbose=False)[0]
    best = None
    for b in r.boxes:
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
        area = (x2 - x1) * (y2 - y1)
        if best is None or area > best[4]:
            best = (x1, y1, x2, y2, area, float(b.conf))
    if best is None:
        return None, None
    x1, y1, x2, y2, _, conf = best
    off = ((x1 + x2) / 2 / frame.shape[1] - 0.5) * fov
    return off, (x1, y1, x2, y2, conf)

def overhead_scene(marker_model, frame, want):
    dets = detect(marker_model, frame)
    return scene_angles(dets, want), dets

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--over", type=int, default=1, help="overhead camera index")
    p.add_argument("--person", type=int, default=0, help="person camera index")
    p.add_argument("--fov", type=float, default=FOV_DEG)
    args = p.parse_args()

    marker_model = load_model()
    from ultralytics import YOLO
    person_model = YOLO("yolov8n.pt")

    cam_over, cam_person = open_cam(args.over), open_cam(args.person)
    board = None
    sign = 1
    status, status_until = "", 0.0
    key_buf = []

    def target_angle(a_scene, off):
        """Person's angle in the overhead frame: X anchor + webcam offset."""
        return a_scene["x"] + sign * off

    def composite(f_over, f_person, dets, pbox, aim=None):
        for name, (x, y, c) in {n: v for n, v in dets.items()}.items():
            cv2.circle(f_over, (int(x), int(y)), 14, BOX[name], 3)
            cv2.putText(f_over, name, (int(x) + 16, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, BOX[name], 2)
        if pbox:
            x1, y1, x2, y2, conf = pbox
            cv2.rectangle(f_person, (x1, y1), (x2, y2), (0, 220, 0), 3)
            cv2.putText(f_person, f"person {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)
        h = 540
        o = cv2.resize(f_over, (int(f_over.shape[1] * h / f_over.shape[0]), h))
        q = cv2.resize(f_person, (int(f_person.shape[1] * h / f_person.shape[0]), h))
        canvas = cv2.hconcat([o, q])
        hud = status if time.time() < status_until else \
            "deliver to person: [k]black [r]red [b]blue [s]silver | [f]lip [q]uit"
        cv2.putText(canvas, hud, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2)
        if aim is not None:
            cv2.putText(canvas, f"target {aim:+.1f} deg", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow("snack rotator duo", canvas)

    def observe(want):
        f_over = read_valid(cam_over)
        f_person = read_valid(cam_person)
        if f_over is None or f_person is None:
            return None, None, None, None
        cv2.imwrite(os.path.expanduser("~/snack-rotator/last_over.jpg"), f_over)
        cv2.imwrite(os.path.expanduser("~/snack-rotator/last_person.jpg"), f_person)
        (a, missing), dets = overhead_scene(marker_model, f_over, want)
        off, pbox = person_offset(person_model, f_person, args.fov)
        composite(f_over, f_person, dets, pbox)
        key_buf.append(cv2.waitKey(1) & 0xFF)   # don't swallow keypresses
        return a, missing, off, pbox

    def deliver(want):
        nonlocal board
        a, missing, off, _ = observe(want)
        if a is None:
            return f"missing markers: {missing}"
        if off is None:
            return "no person visible in webcam"
        if board is None:
            board = make_board()
        cur = board.angle()
        nudge = 15 if cur <= board.hi - 15 else -15
        board.goto(cur + nudge)
        a2, _, _, _ = observe(want)
        if a2 is None:
            return "lost markers during calibration"
        moved = needed_rotation({"x": a2["tape"], "tape": a["tape"]})
        k = moved / nudge
        if abs(k) < 0.3:
            return "disc not following servo"
        for _ in range(4):
            a, missing, off, _ = observe(want)
            if a is None or off is None:
                return "lost person or markers mid-delivery"
            rot = needed_rotation({"tape": a["tape"], "x": target_angle(a, off)})
            if abs(rot) <= TOL_DEG:
                board.cmd("REL")
                return f"{want} delivered to person (err {rot:+.1f})"
            tgt = board.angle() + rot / k
            if not board.lo <= tgt <= board.hi:
                board.cmd("REL")
                return f"person out of reachable arc (servo {tgt:.0f})"
            board.goto(tgt)
        board.cmd("REL")
        return f"person kept moving, residual {rot:+.1f} deg"

    print("duo running - overhead + person cams. focus window for keys")
    while True:
        _ = observe(None)
        keys = [k for k in key_buf if k != 255]
        key_buf.clear()
        if ord("q") in keys:
            break
        if ord("f") in keys:
            sign = -sign
            status, status_until = "webcam direction flipped", time.time() + 3
            print("direction flipped")
        hit = next((k for k in keys if k in KEYS), None)
        if hit is not None:
            print(f"key: deliver {KEYS[hit]}")
            status, status_until = f"delivering {KEYS[hit]} to person...", time.time() + 3600
            status = deliver(KEYS[hit])
            status_until = time.time() + 6
            print(status)
    cam_over.release(); cam_person.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

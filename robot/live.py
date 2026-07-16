#!/usr/bin/env python3
"""Live detection viewer + delivery console for the snack rotator.

    python3 live.py            # camera 1 (iPhone) by default
    python3 live.py --cam 0

Window keys:
    k = deliver black    r = deliver red
    b = deliver blue     s = deliver silver
    q = quit
"""
import argparse, math, os, sys, time
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deliver import load_model, detect, scene_angles, needed_rotation, make_board, TAPES, TOL_DEG

BOX = {"black": (60, 60, 60), "red": (0, 0, 230), "blue": (230, 130, 0),
       "silver": (200, 200, 200), "x": (0, 220, 0)}
KEYS = {ord("k"): "black", ord("r"): "red", ord("b"): "blue", ord("s"): "silver"}

def annotate(frame, model, want=None, status=""):
    r = model.predict(frame, device="cpu", conf=0.35, imgsz=512, verbose=False)[0]
    dets = {}
    for bx in r.boxes:
        name = model.names[int(bx.cls)]
        conf = float(bx.conf)
        x1, y1, x2, y2 = (int(v) for v in bx.xyxy[0].tolist())
        if name not in dets or conf > dets[name][4]:
            dets[name] = (x1, y1, x2, y2, conf)
    for name, (x1, y1, x2, y2, conf) in dets.items():
        c = BOX[name]
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 3)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 3)
    # disc centre + aim lines
    pts = {n: ((v[0] + v[2]) // 2, (v[1] + v[3]) // 2) for n, v in dets.items()}
    tapes = [pts[t] for t in TAPES if t in pts]
    if len(tapes) >= 3:
        cx = int(sum(p[0] for p in tapes) / len(tapes))
        cy = int(sum(p[1] for p in tapes) / len(tapes))
        cv2.circle(frame, (cx, cy), 12, (0, 255, 255), 3)
        if "x" in pts:
            cv2.line(frame, (cx, cy), pts["x"], (0, 220, 0), 2)
        if want and want in pts:
            cv2.line(frame, (cx, cy), pts[want], BOX[want], 2)
    hud = status or "keys: [k]black [r]red [b]blue [s]silver [q]uit"
    cv2.putText(frame, hud, (20, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    return frame, dets

def deliver(model, cam, board, want, show):
    """Closed-loop delivery, keeps the window updating between moves."""
    def observe():
        ok, f = cam.read()
        a, missing = scene_angles(
            {n: ((v[0]+v[2])/2, (v[1]+v[3])/2, v[4]) for n, v in show(f, want)[1].items()}
            if ok else {}, want) if ok else (None, ["camera"])
        return a

    a = observe()
    if a is None:
        return "can't see all markers - fix framing"
    cur = board.angle()
    nudge = 15 if cur <= board.hi - 15 else -15
    board.goto(cur + nudge)
    a2 = observe()
    if a2 is None:
        return "lost markers during calibration"
    moved = needed_rotation({"x": a2["tape"], "tape": a["tape"]})
    k = moved / nudge
    if abs(k) < 0.3:
        return "disc not following servo - check mounting"
    for _ in range(3):
        a = observe()
        if a is None:
            return "lost markers"
        rot = needed_rotation(a)
        if abs(rot) <= TOL_DEG:
            board.cmd("REL")
            return f"{want} delivered (err {rot:+.1f} deg)"
        tgt = board.angle() + rot / k
        if not board.lo <= tgt <= board.hi:
            board.cmd("REL")
            return f"unreachable (needs servo {tgt:.0f}) - the 180deg curse"
        board.goto(tgt)
    board.cmd("REL")
    return f"gave up, residual {rot:+.1f} deg"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=1)
    args = p.parse_args()

    model = load_model()
    cam = cv2.VideoCapture(args.cam)
    assert cam.isOpened(), f"camera {args.cam} not available"
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    board = None
    status, status_until, want = "", 0, None

    def show(frame, w=None):
        f, dets = annotate(frame, model, w, status if time.time() < status_until else "")
        disp = cv2.resize(f, (f.shape[1] * 2 // 3, f.shape[0] * 2 // 3))
        cv2.imshow("snack rotator", disp)
        return f, dets

    print("live view running - focus the window for keys")
    while True:
        ok, frame = cam.read()
        if not ok:
            continue
        show(frame, want)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in KEYS:
            want = KEYS[key]
            status, status_until = f"delivering {want}...", time.time() + 3600
            show(frame, want)
            cv2.waitKey(1)
            try:
                if board is None:
                    board = make_board()
                status = deliver(model, cam, board, want, show)
            except Exception as e:
                status = f"error: {e}"
            status_until = time.time() + 6
            print(status)
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

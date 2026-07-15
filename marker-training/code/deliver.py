#!/usr/bin/env python3
"""Snack rotator Stage 6: rotate a requested tape marker toward the X.

Live-camera closed loop:
    python3 deliver.py red              # camera 0 (Continuity Camera / webcam)
    python3 deliver.py blue --cam 1     # pick a different camera

Single-photo open loop (no camera; uses one image, assumes servo+ = image CCW,
pass --dir -1 if the disc turns the other way):
    python3 deliver.py silver --image ~/Downloads/IMG_6002.HEIC [--dry-run]

Classes: black red blue silver   (x is the target)
"""
import argparse, math, os, subprocess, sys, tempfile, time

MODEL = os.path.expanduser("~/snack-rotator/runs/marker_v3/weights/best.pt")

def find_port():
    import glob as _g
    ports = _g.glob("/dev/cu.usbmodem*")
    if not ports:
        sys.exit("no ESP32 found - plug the board in (/dev/cu.usbmodem*)")
    return ports[0]

PORT = None  # resolved at Board() init
TAPES = ["black", "red", "blue", "silver"]
TOL_DEG = 8          # acceptable alignment error
CONF = 0.35

# ---------- perception ----------

def load_model():
    from ultralytics import YOLO
    return YOLO(MODEL)

CLASS_CONF = {"x": 0.20, "black": 0.30, "red": 0.35, "blue": 0.35, "silver": 0.30}

def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ar = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ar if ar else 0

def detect(model, img):
    """Return {name: (cx, cy, conf)}: best per class, cross-class overlaps suppressed."""
    r = model.predict(img, device="cpu", conf=0.15, verbose=False)[0]
    cands = []
    for b in r.boxes:
        name = model.names[int(b.cls)]
        conf = float(b.conf)
        if conf < CLASS_CONF.get(name, CONF):
            continue
        cands.append((conf, name, b.xyxy[0].tolist()))
    cands.sort(reverse=True)
    kept = []
    for conf, name, box in cands:
        if any(_iou(box, kb) > 0.6 for _, _, kb in kept):
            continue  # two classes claiming the same patch: trust the stronger
        kept.append((conf, name, box))
    out = {}
    for conf, name, (x1, y1, x2, y2) in kept:
        if name not in out or conf > out[name][2]:
            out[name] = ((x1 + x2) / 2, (y1 + y2) / 2, conf)
    return out

def scene_angles(dets, want):
    """Angles (deg, image CCW, y-down corrected) of tape `want` and x around disc centre."""
    tapes = [dets[t] for t in TAPES if t in dets]
    missing = [t for t in TAPES + ["x"] if t not in dets]
    if want not in dets or "x" not in dets or len(tapes) < 3:
        return None, missing
    cx = sum(t[0] for t in tapes) / len(tapes)
    cy = sum(t[1] for t in tapes) / len(tapes)
    def ang(p):  # image y grows down; negate for math-CCW
        return math.degrees(math.atan2(-(p[1] - cy), p[0] - cx))
    return {"tape": ang(dets[want]), "x": ang(dets["x"]),
            "centre": (cx, cy), "conf": dets[want][2]}, missing

def needed_rotation(a):
    """Smallest CCW-positive rotation that brings tape onto x, in (-180, 180]."""
    d = (a["x"] - a["tape"]) % 360
    return d - 360 if d > 180 else d

# ---------- actuation ----------

class EV3Board:
    """LEGO EV3 Large motor backend: unlimited rotation, 1-deg encoder."""
    lo, hi = -10**9, 10**9

    def __init__(self):
        import ev3_dc as ev3
        self._brick = ev3.EV3(protocol=ev3.USB)
        for name, port in (("A", ev3.PORT_A), ("B", ev3.PORT_B),
                           ("C", ev3.PORT_C), ("D", ev3.PORT_D)):
            try:
                self.m = ev3.Motor(port, ev3_obj=self._brick)
                print(f"EV3 motor on port {name}")
                break
            except Exception:
                continue
        else:
            raise RuntimeError("EV3 brick found but no motor on A-D")

    def angle(self):
        return self.m.position

    def goto(self, deg):
        self.m.start_move_to(round(deg), speed=25, brake=True)
        while self.m.busy:
            time.sleep(0.05)
        time.sleep(0.25)
        return round(deg)

    def cmd(self, c):
        if c == "REL":
            self.m.stop(brake=False)
            return "RELEASED"
        return "PONG" if c == "PING" else "DONE"


def make_board():
    """Prefer the EV3 (unlimited rotation); fall back to the ESP32 servo."""
    try:
        return EV3Board()
    except Exception as e:
        print(f"(no EV3: {e}) - using ESP32 serial servo")
        return Board()


class Board:
    lo, hi = 0, 180

    def __init__(self, port=None):
        import serial
        port = port or find_port()
        print(f"board on {port}")
        self.ser = serial.Serial(port, 115200, timeout=3)
        self.ser.dtr = True
        time.sleep(0.8)
        self.ser.reset_input_buffer()
        assert self.cmd("PING") == "PONG", "board not responding"

    def cmd(self, c):
        self.ser.write((c + "\n").encode())
        while True:
            line = self.ser.readline().decode(errors="replace").strip()
            if not line:
                return None
            if line.startswith(("DONE", "POS", "PONG", "RELEASED", "ERR")):
                return line

    def angle(self):
        return int(self.cmd("WHERE").split("angle=")[1].split()[0])

    def goto(self, deg):
        deg = max(0, min(180, round(deg)))
        self.cmd(f"GOTO {deg}")
        time.sleep(0.4)
        return deg

# ---------- image sources ----------

def heic_safe(path):
    if path.lower().endswith((".heic", ".heif")):
        out = tempfile.mktemp(suffix=".jpg")
        subprocess.run(["sips", "-s", "format", "jpeg", path, "--out", out],
                       capture_output=True, check=True)
        return out
    return path

class Camera:
    def __init__(self, index):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        assert self.cap.isOpened(), f"camera {index} not available"
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    def grab(self):
        for _ in range(5):        # flush stale buffered frames
            self.cap.read()
        ok, frame = self.cap.read()
        assert ok, "camera read failed"
        self.cv2.imwrite(os.path.expanduser("~/snack-rotator/last_frame.jpg"), frame)
        return frame

# ---------- main flows ----------

def flow_image(model, want, path, direction, dry):
    dets = detect(model, heic_safe(os.path.expanduser(path)))
    a, missing = scene_angles(dets, want)
    if a is None:
        sys.exit(f"can't plan: missing detections {missing}")
    rot = needed_rotation(a)
    print(f"{want} at {a['tape']:.1f} deg, X at {a['x']:.1f} deg "
          f"(conf {a['conf']:.2f})  ->  rotate {rot:+.1f} deg (image CCW)")
    if dry:
        return
    b = make_board()
    cur = b.angle()
    tgt = cur + direction * rot
    if not b.lo <= tgt <= b.hi:
        sys.exit(f"target angle {tgt:.0f} outside range - move the X or re-seat the disc")
    print(f"servo {cur} -> {tgt:.0f}")
    b.goto(tgt)
    b.cmd("REL")
    print("done (open loop - snap another photo to verify)")

def flow_live(model, want, cam_index):
    cam = Camera(cam_index)
    b = make_board()

    def observe():
        a, missing = scene_angles(detect(model, cam.grab()), want)
        return a, missing

    a, missing = observe()
    if a is None:
        sys.exit(f"can't see the scene: missing {missing}")
    rot = needed_rotation(a)
    print(f"{want} at {a['tape']:.1f}, X at {a['x']:.1f} -> need {rot:+.1f} deg")

    # calibrate direction: nudge +15 servo deg, see which way the tape moved
    cur = b.angle()
    nudge = 15 if cur <= b.hi - 15 else -15
    b.goto(cur + nudge)
    a2, _ = observe()
    if a2 is None:
        sys.exit("lost markers during calibration nudge")
    moved = needed_rotation({"x": a2["tape"], "tape": a["tape"]})  # tape angle change
    k = moved / nudge  # image-deg per servo-deg (sign + coupling ratio)
    print(f"calibration: {nudge:+d} servo deg moved tape {moved:+.1f} img deg (k={k:.2f})")
    if abs(k) < 0.3:
        sys.exit("disc doesn't seem to follow the servo - check mounting")

    for attempt in range(3):
        a, missing = observe()
        if a is None:
            sys.exit(f"lost markers: {missing}")
        rot = needed_rotation(a)
        print(f"attempt {attempt + 1}: error {rot:+.1f} deg")
        if abs(rot) <= TOL_DEG:
            print(f"aligned within {TOL_DEG} deg - delivered!")
            b.goto(b.angle())
            b.cmd("REL")
            return
        cur = b.angle()
        tgt = cur + rot / k
        if not b.lo <= tgt <= b.hi:
            sys.exit(f"target {tgt:.0f} outside motor range - move the X or re-seat the disc")
        b.goto(tgt)
    print(f"stopped after 3 attempts, residual error {rot:+.1f} deg")
    b.cmd("REL")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("snack", choices=TAPES)
    p.add_argument("--image", help="single photo instead of live camera")
    p.add_argument("--cam", type=int, default=0, help="camera index (live mode)")
    p.add_argument("--dir", type=int, default=1, choices=(1, -1),
                   help="image-mode only: +1 if servo+ turns disc CCW in image")
    p.add_argument("--dry-run", action="store_true", help="image mode: plan only, no serial")
    args = p.parse_args()

    model = load_model()
    if args.image:
        flow_image(model, args.snack, args.image, args.dir, args.dry_run)
    else:
        flow_live(model, args.snack, args.cam)

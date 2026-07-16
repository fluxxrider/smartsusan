#!/usr/bin/env python3
"""LLM-vision snack delivery: Claude looks at the overhead photo, finds the
requested snack + the X target + the platter centre; the EV3 rotates to align.

    export ANTHROPIC_API_KEY=sk-ant-...
    python3 llm_deliver.py welchs            # camera 0 (Logitech)
    python3 llm_deliver.py bear --cam 1

No trained model needed. Each observation is one Claude vision call (~3-8s).
"""
import argparse, base64, math, os, sys, time

import cv2
import anthropic

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deliver import make_board

SNACKS = ["bar", "bear", "juice", "welchs"]
DESCRIPTIONS = {
    "bar": "the NutriGrain breakfast bar (blue wrapper)",
    "bear": "the BEAR fruit splits package (green/orange wrapper)",
    "juice": "the CapriSun juice pouch (silver pouch with straw)",
    "welchs": "the Welch's fruit snacks packet (blue packet)",
}
TOL_DEG = 12
MAX_TRIES = 8
DAMP = 0.7          # apply 70% of computed correction (kills overshoot oscillation)
MAX_STEP = 90       # never rotate more than this many platter-deg in one move

SCHEMA = {
    "type": "object",
    "properties": {
        "platter_center": {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"], "additionalProperties": False,
        },
        "item_found": {"type": "boolean"},
        "item": {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"], "additionalProperties": False,
        },
        "x_marker_found": {"type": "boolean"},
        "x_marker": {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"], "additionalProperties": False,
        },
    },
    "required": ["platter_center", "item_found", "item", "x_marker_found", "x_marker"],
    "additionalProperties": False,
}

client = anthropic.Anthropic()
_FAST_OK = True
_LOCAL = None


def local_models():
    global _LOCAL
    if _LOCAL is None:
        from deliver import load_model
        _LOCAL = load_model(snacks=True)
    return _LOCAL


def observe_local(frame, want):
    """Fast local detection; returns pixels or None (no centre - anchors supply it)."""
    from deliver import detect
    dets = detect(local_models(), frame)
    if want in dets and "x" in dets:
        return {"centre": None,
                "item": (dets[want][0], dets[want][1]),
                "x": (dets["x"][0], dets["x"][1])}
    return None


def grab(cap):
    for _ in range(5):
        cap.read()
    ok, frame = cap.read()
    assert ok, "camera read failed"
    h, w = frame.shape[:2]
    if w > 1024:
        frame = cv2.resize(frame, (1024, int(h * 1024 / w)))
    cv2.imwrite(os.path.expanduser("~/snack-rotator/llm_last.jpg"), frame)
    return frame


def observe(cap, want, anchors=None, prefer_local=True):
    frame = grab(cap)
    if prefer_local and anchors is not None and anchors.centre is not None:
        o = observe_local(frame, want)
        if o is not None:
            print(f"  [local] item=({o['item'][0]:.0f},{o['item'][1]:.0f}) x=({o['x'][0]:.0f},{o['x'][1]:.0f})")
            return o
        print("  (local model came up empty - asking claude)")
    return observe_llm(frame, want)


def observe_llm(frame, want):
    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.standard_b64encode(jpg.tobytes()).decode()

    t0 = time.time()
    def _call(fast):
        kw = dict(speed="fast", betas=["fast-mode-2026-02-01"]) if fast else {}
        api = client.beta.messages if fast else client.messages
        return api.create(
            model="claude-opus-4-8",
            max_tokens=2048,
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            **kw,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text":
                    "This is an overhead photo of a cardboard rotating platter with snacks on it, "
                    "and a separate white paper sheet with a large hand-drawn X, lying on the "
                    "table NEXT TO the platter.\n"
                    f"Locate, in pixel coordinates of this image:\n"
                    f"1. platter_center: the centre of the round cardboard platter\n"
                    f"2. item: the centre of {DESCRIPTIONS[want]} - it is ON the round cardboard "
                    f"platter. IGNORE any snack packets lying elsewhere on the desk; only report "
                    f"the one on the platter (set item_found false if not visible)\n"
                    f"3. x_marker: the centre of the X drawn on the WHITE PAPER SHEET beside the platter. "
                    f"IMPORTANT: the cardboard platter has sector lines drawn on it that cross at its "
                    f"centre - that is NOT the x_marker. The x_marker is always OFF the platter, on white "
                    f"paper. (set x_marker_found false if not visible)"},
            ],
        }],
        )
    global _FAST_OK
    if _FAST_OK:
        try:
            response = _call(fast=True)
        except Exception as e:
            print(f"  (fast mode unavailable: {type(e).__name__} - standard from now on)")
            _FAST_OK = False
            response = _call(fast=False)
    else:
        response = _call(fast=False)
    import json
    data = json.loads(next(b.text for b in response.content if b.type == "text"))
    print(f"  [claude {time.time()-t0:.1f}s] centre=({data['platter_center']['x']},{data['platter_center']['y']}) "
          f"item={'(%d,%d)' % (data['item']['x'], data['item']['y']) if data['item_found'] else 'NOT FOUND'} "
          f"x={'(%d,%d)' % (data['x_marker']['x'], data['x_marker']['y']) if data['x_marker_found'] else 'NOT FOUND'}")
    if not (data["item_found"] and data["x_marker_found"]):
        return None
    return {"centre": (data["platter_center"]["x"], data["platter_center"]["y"]),
            "item": (data["item"]["x"], data["item"]["y"]),
            "x": (data["x_marker"]["x"], data["x_marker"]["y"])}


class Anchors:
    """The platter centre and the X sheet don't move - anchor them and reject teleports."""
    def __init__(self):
        self.centre = None
        self.x = None

    def angles(self, obs):
        def ema(old, new, alpha=0.3):
            return (old[0]*(1-alpha)+new[0]*alpha, old[1]*(1-alpha)+new[1]*alpha)
        if self.centre is None:
            self.centre, self.x = obs["centre"], obs["x"]
        else:
            if obs["centre"] is not None and math.dist(obs["centre"], self.centre) < 80:
                self.centre = ema(self.centre, obs["centre"])
            if math.dist(obs["x"], self.x) < 80:
                self.x = ema(self.x, obs["x"])
            else:
                print(f"  (rejected X teleport to {obs['x']} - keeping anchor {tuple(round(v) for v in self.x)})")
        cx, cy = self.centre
        def ang(p):
            return math.degrees(math.atan2(-(p[1] - cy), p[0] - cx))
        return {"item": ang(obs["item"]), "x": ang(self.x)}


def needed(a):
    d = (a["x"] - a["item"]) % 360
    return d - 360 if d > 180 else d


CALIB_FILE = os.path.expanduser("~/snack-rotator/calib.json")


def load_k():
    try:
        import json
        return json.load(open(CALIB_FILE))["k"]
    except Exception:
        return None


def save_k(k):
    import json
    json.dump({"k": k}, open(CALIB_FILE, "w"))


def calibrate(cap, board, want, a, anchors):
    cur = board.angle()
    print("calibration nudge +20...")
    board.goto(cur + 20)
    o2 = observe(cap, want, anchors)
    if o2 is None:
        print("lost sight during calibration")
        return None
    a2 = anchors.angles(o2)
    moved = needed({"x": a2["item"], "item": a["item"]})
    k = moved / 20
    print(f"calibration: +20 motor deg moved item {moved:+.1f} deg (k={k:.2f})")
    if abs(k) < 0.1:
        print("platter didn't move - check drivetrain")
        return None
    save_k(k)
    return k


def deliver_one(cap, board, want, k=None, anchors=None):
    """Deliver one snack; returns learned k (deg-per-motor-deg) or None on failure."""
    print(f"\n=== delivering {want} ===")
    anchors = anchors or Anchors()
    obs = observe(cap, want, anchors, prefer_local=False)  # first look via claude (gets centre)
    if obs is None:
        print("can't see the item and the X - fix framing")
        return None
    a = anchors.angles(obs)
    rot = needed(a)
    print(f"{want} at {a['item']:.1f} deg, X at {a['x']:.1f} deg -> need {rot:+.1f} deg")

    if k is None:
        k = calibrate(cap, board, want, a, anchors)
        if k is None:
            return None
        obs = observe(cap, want, anchors)      # fresh look after the nudge
        if obs is None:
            return None
        a = anchors.angles(obs)
        rot = needed(a)

    for attempt in range(MAX_TRIES):
        if abs(rot) <= TOL_DEG:
            print(f"DELIVERED - {want} aligned with X (err {rot:+.1f} deg)")
            save_k(k)
            return k
        step = max(-MAX_STEP, min(MAX_STEP, rot * DAMP))
        motor_delta = step / k
        board.goto(board.angle() + motor_delta)
        prev_rot = rot
        expected = prev_rot - step   # where the error should land if the move was perfect
        obs = observe(cap, want, anchors)      # verify after the move
        if obs is None:
            print("lost sight mid-delivery")
            return None
        a = anchors.angles(obs)
        rot = needed(a)
        if abs(rot - expected) > 70:           # item "teleported" - probably a misdetection
            print(f"  (implausible jump: expected ~{expected:+.0f}, saw {rot:+.0f} - second opinion)")
            obs = observe(cap, want, anchors, prefer_local=False)
            if obs is None:
                print("lost sight mid-delivery")
                return None
            a = anchors.angles(obs)
            rot = needed(a)
        # re-learn k from what actually happened: platter moved (prev_rot - rot)
        # image-deg for motor_delta motor-deg
        k_meas = (prev_rot - rot) / motor_delta if abs(motor_delta) > 2 else None
        if (k_meas is not None and (k_meas > 0) == (k > 0)
                and abs(k) / 3 < abs(k_meas) < abs(k) * 3):
            k = 0.6 * k + 0.4 * k_meas
            print(f"attempt {attempt+1}: error {rot:+.1f} deg  (k updated to {k:.2f})")
        else:
            print(f"attempt {attempt+1}: error {rot:+.1f} deg")
    print(f"gave up on {want} after {MAX_TRIES} attempts, residual {rot:+.1f} deg")
    save_k(k)
    return k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("snacks", nargs="+", choices=SNACKS,
                   help="one or more snacks, delivered in order")
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--recal", action="store_true", help="force fresh gear-ratio calibration")
    p.add_argument("--pause", type=float, default=3.0,
                   help="seconds to hold at the X between deliveries")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.cam)
    assert cap.isOpened(), f"camera {args.cam} not available"
    board = make_board()

    k = None if args.recal else load_k()
    if k:
        print(f"using cached calibration k={k:.2f} (pass --recal after changing the drivetrain)")
    anchors = Anchors()
    for i, want in enumerate(args.snacks):
        k = deliver_one(cap, board, want, k, anchors)
        if k is None:
            break
        if i < len(args.snacks) - 1:
            print(f"holding {args.pause}s...")
            time.sleep(args.pause)
    board.cmd("REL")
    print("\nsequence complete - motor released")


if __name__ == "__main__":
    main()

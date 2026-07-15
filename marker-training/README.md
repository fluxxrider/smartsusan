# smartsusan 🥨

Autonomous rotating snack platform: an overhead camera finds a requested snack
marker and the person who asked for it, then a LEGO EV3 motor rotates the
platter to deliver it. Vision = YOLOv8n, five classes: `black red blue silver`
(snack markers) + `x` (delivery target).

## FOR THE FRIEND WITH THE 5060 — do this

```bash
git clone https://github.com/polarizedfortnite-cpu/smartsusan
cd smartsusan
pip install ultralytics
python bootstrap.py
```

That's it. ~10 minutes on a 5060. It trains a seed model on the small clean
dataset, uses it to auto-label all 458 raw frames, then trains the real model
on the expanded set and prints per-class accuracy.

**Send back the file `runs/round2/weights/best.pt`** (Discord/Drive/whatever).

## What's in here

| Path | What |
|---|---|
| `dataset/` | 39 hand-audited seed frames (YOLO format) |
| `raw_frames/` | all 458 captured frames from the actual demo camera |
| `bootstrap.py` | two-round train script (see above) |
| `marker_v3_seed.pt` | previous-generation model (old small platter) |
| `code/` | the robot itself: delivery loop, live console, dual-camera person tracking, EV3 throttle |

## The robot (for context)

- `code/deliver.py` — closed loop: observe → compute rotation → move EV3 → verify (±8°)
- `code/live.py` — live detection viewer + keyboard delivery console
- `code/duo.py` — dual camera: overhead finds markers, webcam finds the PERSON; delivers to them
- `code/throttle.py` — interactive continuous-rotation speed control
- Actuator: LEGO EV3 Large motor over USB (`ev3_dc`), unlimited rotation, 1° encoder

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

"Smart Susan" — an autonomous rotating snack platform (Tufts AI final project, demo already shipped). An overhead camera finds a requested snack on a rotating cardboard platter plus an X delivery target beside it; a LEGO EV3 motor rotates the platter until the snack aligns with the X. Perception is a hybrid: local YOLOv8 models when they work, Claude vision (`claude-opus-4-8` with JSON-schema output) as fallback. There are no tests, linters, or CI — this is demo-grade robotics code.

## Repository layout

- `robot/` — **the canonical, latest robot code** (demo-day state). All runtime scripts live here. `robot/DEMO.md` is the demo runbook and troubleshooting table.
- `robot/app/` — single-file voice web app (`index.html`): ElevenLabs realtime STT/TTS, posts snack orders to the snack server. Config (ElevenLabs key placeholder, robot server IP) is in constants at the top of the `<script>` block.
- `marker-training/` — self-contained training bundle for the **marker detector**, written to be handed to a friend with a GPU (see its README). `bootstrap.py` does two-round training: seed YOLOv8n on the 39 hand-audited frames in `dataset/`, pseudo-label all 458 `raw_frames/`, retrain → `runs/round2/weights/best.pt`.
- `train-scripts/` — Arch Linux + CUDA notebook pipeline (`00_…` through `05_…` in order) for the **snack detector**, plus `train_full.py` (standalone 60-epoch YOLOv8s run, easier to babysit via log file than a notebook) and `run_live_inference.py` (macOS live-webcam check, `mps` device).
- `dataset/` — ~4 GB of raw snack photos: `<snack>-<n>/` and background-removed `<snack>-<n>-nobg/` pairs, plus tray backgrounds `bg1`–`bg4`.
- `dev/programming/2026/Tufts-ai-final/` — earliest iteration of the pipeline (scraper → bg removal → synth dataset → train) as interactive scripts targeting Apple `mps`. Historical; superseded by `train-scripts/`.

### Duplication warning

`marker-training/code/` and `marker-training/synth/synth_gen.py` are frozen snapshots of `robot/` scripts for the GPU-friend handoff. Most are byte-identical, but `marker-training/code/deliver.py` is an **older marker-only version** of `robot/deliver.py` (no snack-model support). Make changes in `robot/`; only touch the `marker-training/` copies if intentionally refreshing the handoff bundle.

## Two detector families — don't mix the class maps

1. **Marker detector** (YOLOv8n): classes `0=black 1=red 2=blue 3=silver 4=x` (tape markers on the old platter). Weights: `marker-training/marker_v3_seed.pt`, `marker-training/synth/marker_v4.pt`.
2. **Snack detector** (YOLOv8s): classes `0=bar 1=bear 2=juice 3=welchs 4=x` (real snack packets: NutriGrain bar, BEAR splits, CapriSun, Welch's). Weights: `train-scripts/snacks_best.pt`, `train-scripts/snacks_best_singleobject_60ep.pt`, `robot/models/snacks_unified.pt` (all 5 classes, trained on the real rig — preferred), `robot/models/best.pt`.

`x` (the delivery target drawn on a paper sheet **off** the platter) appears in both. Class ID order is baked into every `data.yaml`, `autolabel.py`, `bootstrap.py`, and `synth_gen.py` — keep it consistent.

## Commands

### Running the robot (macOS deploy machine)

Robot scripts hard-code `~/snack-rotator/` for weights, `calib.json`, and snapshot output — on the demo machine the code lives there, not in this repo checkout. `deliver.py` expects weights at `~/snack-rotator/runs/marker_v4/best.pt`, `~/snack-rotator/snacks_best.pt`, and (optionally) `~/snack-rotator/runs/snacks_unified.pt`. `llm_deliver.py` and `snack_server.py` need `ANTHROPIC_API_KEY` exported.

```bash
python3 snack_server.py --cam 1          # HTTP bridge on :5050 (POST /deliver, GET /status, GET /snacks)
python3 llm_deliver.py welchs --cam 1    # direct delivery; --recal redoes gear calibration, --pause N holds at X
cd robot/app && npm run dev              # voice app (python http.server on :5173)
python3 list_cams.py                     # probe camera indices 0-4, save snapshots
python3 frame_view.py --cam 1            # live aiming view (d = detection overlay, q = quit)
python3 live.py --cam 1                  # detection viewer + keyboard delivery console
python3 throttle.py                      # manual EV3 speed control (+/- r 0 q)
```

### Training

```bash
# Marker detector (any CUDA box; ~10 min on a 5060)
cd marker-training && pip install ultralytics && python bootstrap.py
# → send back runs/round2/weights/best.pt

# Snack detector (Arch + CUDA box; deps live in the shared venv ~/pytorch-venv,
# NOT a project-local one — see train-scripts/pyproject.toml comments and 00_environment_setup.ipynb)
cd train-scripts && python train_full.py     # 60-epoch YOLOv8s on snacks_yolo_dataset/

# Synthetic snack dataset from real rig frames (crops pasted onto backgrounds, X auto-labeled by marker_v4)
cd robot && python3 synth_gen.py

# Live sanity check of a trained model (macOS)
cd train-scripts && python run_live_inference.py --model snacks_best.pt   # --source clip.mp4 for a video file
```

GPU note: if PyPI `torch` doesn't detect an RTX 5060 (sm_120/Blackwell), reinstall from the CUDA index — instructions in `train-scripts/pyproject.toml`.

## Architecture: the delivery loop

`robot/deliver.py` is the shared library everything imports (scripts do `sys.path.insert(0, <own dir>)` — there's no package structure).

1. **Perception** — `detect()` runs one or more YOLO models on CPU, merges detections, applies per-class confidence floors (`CLASS_CONF`), suppresses cross-class overlaps by IoU, returns best-per-class centers.
2. **Geometry** — angles are computed image-CCW (y negated) around the platter centre; `needed_rotation()` returns the smallest signed rotation in (−180, 180].
3. **Actuation** — `make_board()` prefers `EV3Board` (LEGO EV3 Large motor over USB via `ev3_dc`, unlimited rotation, 1° encoder) and falls back to `Board` (ESP32 servo over serial at 115200, text protocol `PING/GOTO/WHERE/REL`, limited to 0–180°).
4. **Closed loop** (`llm_deliver.py`) — observe → rotate → verify until within `TOL_DEG`. Corrections are damped (`DAMP = 0.7`) and capped (`MAX_STEP = 90`) to kill overshoot oscillation. The `Anchors` class EMA-smooths the platter centre and X position and **rejects teleports** (>80 px jumps), since the platter and X sheet physically never move. A gear-ratio constant `k` (motor-deg per platter-deg) is learned on the fly and persisted to `~/snack-rotator/calib.json`.
5. **Observation strategy** — try the local snack model first (`observe_local`); if it misses, fall back to a Claude vision call (`observe_llm`, ~3–8 s) using structured output (JSON schema) and the fast-mode beta with graceful degradation. Fresh `Anchors` per order — never trust a stale X.

`snack_server.py` wraps `llm_deliver` in a `ThreadingHTTPServer` with a busy-lock (409 while a delivery runs); the voice app is just a client of it.

## Conventions and gotchas

- Plain scripts with docstring-as-usage headers, `argparse`, compact style. Inference at runtime is CPU (`device="cpu"`); training uses CUDA, `mps` on Macs.
- macOS-specifics in robot code: `/dev/cu.usbmodem*` for the ESP32, `sips` for HEIC conversion, Continuity Camera / camera index shuffling (hence `list_cams.py`).
- `train-scripts/.gitignore` excludes `runs/`, generated datasets, and `*.pt` — the committed weight files were added deliberately. Large binaries (weights, the 4 GB `dataset/`) are committed on purpose; don't "clean them up".
- The ElevenLabs key in `robot/app/index.html` is a `PUT_YOUR_ELEVENLABS_KEY_HERE` placeholder and `ROBOT_BASE_URL` is a hardcoded LAN IP — both are edited in place at demo time. Never commit a real key; `robot/DEMO.md` mandates rotating the Anthropic and ElevenLabs keys after each demo.
- Physical staging rules matter more than code (from `DEMO.md`): no spare snacks in camera view (decoys break detection), X flat on white paper beside the platter (never on it — the platter's sector lines cross at its centre and get mistaken for the X), snacks near the rim with wrappers visible, camera fixed once a run starts.

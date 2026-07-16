# Smart Susan — Demo Runbook

## Setup once per session (Terminal)
```bash
export ANTHROPIC_API_KEY=sk-ant-...     # needed in every new terminal window
```

## THE DEMO (voice app mode)
```bash
# 1. start the robot server (owns camera + motor; leave running)
python3 ~/snack-rotator/snack_server.py --cam 1     # --cam 0 if index shifted

# 2. start the app (separate terminal, in the app folder)
cd ~/Documents/miscellaneous/smart-susan && npx vite  # or however it was started
# open http://localhost:5173 in the browser -> tap mic -> "can I get the fruit snacks"
```

## Direct delivery (no app, one command)
```bash
python3 ~/snack-rotator/llm_deliver.py welchs --cam 1          # one snack
python3 ~/snack-rotator/llm_deliver.py welchs bear welchs --cam 1   # sequence
# flags: --recal (redo gear calibration)  --pause 5 (hold at X longer)
```

## Aiming / checking the camera
```bash
python3 ~/snack-rotator/list_cams.py         # who is on which index (saves snapshots)
python3 ~/snack-rotator/frame_view.py --cam 1  # live view; q quits
```

## Manual motor control
```bash
python3 ~/snack-rotator/throttle.py          # +/- speed, r reverse, 0 stop, q quit
```

## If things break
| Symptom | Fix |
|---|---|
| "camera N not available" | list_cams.py to find the right index; Logi unplugged? iPhone: screen OFF on mount |
| App says "not connected" | snack_server not running, or crashed — restart it |
| "no motor found on A-D" | EV3 brick on? USB in? cable clicked into a LETTERED port? |
| Deliveries oscillate/miss | clear ALL spare snacks off the desk (decoys!); re-tape snacks so wrappers show |
| Aim consistently off | platter/X sheet must be fully in frame; X flat, fat, on white paper OFF the platter |
| Everything weird after rebuild | add --recal once |

## Stage rules (matter more than any code)
- NO spare snack packets anywhere in camera view
- X sheet: flat, large fat X, on the table beside the platter (never on it)
- snacks near the platter RIM, wrappers visible, taped from underneath
- decent light; camera can't move once the run starts

## After the demo
- Terminate any RunPod pods (console — $/hr!)
- Rotate the Anthropic API key (console.anthropic.com) and the ElevenLabs key

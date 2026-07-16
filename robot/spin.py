#!/usr/bin/env python3
"""Spin the snack-rotator disc slowly for training-video capture.

Sweeps 0 -> 180 -> 0 continuously in small steps, pausing briefly at each
position so frames aren't motion-blurred. Ctrl-C stops and releases the servo.

Usage: python3 spin.py [minutes]   (default 1 minute)
"""
import serial, sys, time

import glob
_ports = glob.glob("/dev/cu.usbmodem*")
PORT = _ports[0] if _ports else "/dev/cu.usbmodem101"
STEP_DEG = 5        # degrees per move
DWELL_S = 0.35      # pause at each position (sharp frames at 2 fps extraction)
MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

ser = serial.Serial(PORT, 115200, timeout=3)
ser.dtr = True
time.sleep(1.0)
ser.reset_input_buffer()

def cmd(c):
    ser.write((c + "\n").encode())
    while True:
        line = ser.readline().decode(errors="replace").strip()
        if not line:
            return None
        if line.startswith(("DONE", "POS", "PONG", "RELEASED", "ERR")):
            return line

assert cmd("PING") == "PONG", "board not responding"
print(f"Spinning for {MINUTES} min — start filming now.")

deadline = time.time() + MINUTES * 60
angle, direction = 0, 1
cmd("GOTO 0")
try:
    while time.time() < deadline:
        angle += STEP_DEG * direction
        if angle >= 180: angle, direction = 180, -1
        elif angle <= 0: angle, direction = 0, 1
        cmd(f"GOTO {angle}")
        print(f"\r  angle {angle:3d}  {int(deadline - time.time()):3d}s left ", end="", flush=True)
        time.sleep(DWELL_S)
finally:
    cmd("GOTO 90")
    cmd("REL")
    ser.close()
    print("\nDone — servo released at 90.")

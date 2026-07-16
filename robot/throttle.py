#!/usr/bin/env python3
"""Continuous-rotation throttle for the EV3 platter drive.

    python3 throttle.py

Keys (no Enter needed):
    + / =  faster        - :  slower
    r      reverse       0 / space :  stop
    q      quit (motor released)
"""
import sys, termios, time, tty, select
import ev3_dc as ev3

brick = ev3.EV3(protocol=ev3.USB)
motor = None
for name, port in (("A", ev3.PORT_A), ("B", ev3.PORT_B), ("C", ev3.PORT_C), ("D", ev3.PORT_D)):
    try:
        motor = ev3.Motor(port, ev3_obj=brick)
        print(f"motor on port {name}")
        break
    except Exception:
        pass
assert motor, "no motor found on A-D"

speed = 0          # -100 .. 100 (%)
STEP = 5

def apply(sp):
    if sp == 0:
        motor.stop(brake=False)
    else:
        motor.start_move(speed=abs(sp), direction=1 if sp > 0 else -1)

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
tty.setcbreak(fd)
print("throttle ready:  +/- speed | r reverse | 0 stop | q quit")
try:
    while True:
        if select.select([sys.stdin], [], [], 0.15)[0]:
            c = sys.stdin.read(1)
            if c == "q":
                break
            elif c in "+=":
                speed = min(100, speed + STEP)
            elif c == "-":
                speed = max(-100, speed - STEP)
            elif c == "r":
                speed = -speed
            elif c in "0 ":
                speed = 0
            apply(speed)
        bar = "#" * (abs(speed) // 5)
        sys.stdout.write(f"\rspeed {speed:+4d}%  |{bar:<20s}|  encoder {motor.position:>7d} deg   ")
        sys.stdout.flush()
finally:
    motor.stop(brake=False)
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    print("\nstopped, motor released")

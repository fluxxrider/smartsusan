#!/usr/bin/env python3
"""HTTP bridge: phones (or anything) request snacks; this Mac delivers them.

    export ANTHROPIC_API_KEY=sk-ant-...
    python3 snack_server.py            # serves on 0.0.0.0:5050, camera 1

    POST /deliver  {"snack": "welchs"}   -> {"status": "delivering"}
    GET  /status                          -> {"busy": false, "last": "welchs: delivered"}
    GET  /snacks                          -> {"snacks": ["bar","bear","juice","welchs"]}

Phone app on the same Wi-Fi calls http://<this-mac-ip>:5050.
"""
import argparse, json, sys, os, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_deliver as L
from deliver import make_board

state = {"busy": False, "last": "idle"}
lock = threading.Lock()
cap = None
board = None
anchors = None
k = None


def do_delivery(snack):
    global k, anchors
    state["busy"] = True
    state["last"] = f"{snack}: delivering"
    anchors = L.Anchors()   # fresh eyes every order - never trust a stale X
    try:
        k2 = L.deliver_one(cap, board, snack, k, anchors)
        if k2 is not None:
            k = k2
            state["last"] = f"{snack}: delivered"
        else:
            state["last"] = f"{snack}: failed (couldn't see it)"
    except Exception as e:
        state["last"] = f"{snack}: error {type(e).__name__}"
    finally:
        board.cmd("REL")
        state["busy"] = False


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")   # phone app friendliness
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._json(200, {})

    def do_GET(self):
        if self.path == "/status":
            self._json(200, state)
        elif self.path == "/snacks":
            self._json(200, {"snacks": L.SNACKS})
        else:
            self._json(404, {"error": "unknown path"})

    def do_POST(self):
        if self.path != "/deliver":
            self._json(404, {"error": "unknown path"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            snack = json.loads(self.rfile.read(length))["snack"].lower()
        except Exception:
            self._json(400, {"error": "body must be {\"snack\": \"...\"}"})
            return
        if snack not in L.SNACKS:
            self._json(400, {"error": f"unknown snack, pick from {L.SNACKS}"})
            return
        with lock:
            if state["busy"]:
                self._json(409, {"error": "already delivering", "last": state["last"]})
                return
            threading.Thread(target=do_delivery, args=(snack,), daemon=True).start()
        self._json(200, {"status": "delivering", "snack": snack})

    def log_message(self, *a):
        pass


def main():
    global cap, board, anchors, k
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=1)
    p.add_argument("--port", type=int, default=5050)
    args = p.parse_args()

    cap = cv2.VideoCapture(args.cam)
    assert cap.isOpened(), f"camera {args.cam} not available"
    board = make_board()
    anchors = L.Anchors()
    k = L.load_k()

    import socket
    ip = socket.gethostbyname(socket.gethostname())
    print(f"snack server on http://{ip}:{args.port}  (POST /deliver, GET /status, GET /snacks)")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

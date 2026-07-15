"""Live webcam inference for the snack detector.

Meant to run on the MacBook (this training box's webcam is unreliable) —
auto-detects Apple Silicon `mps` if available, falling back to CUDA or CPU
so the same script also works here for testing against a video file.

Usage:
    python run_live_inference.py --model snacks_best.pt
    python run_live_inference.py --camera 1        # if index 0 isn't the right camera
    python run_live_inference.py --source clip.mp4  # test against a video file instead of a live camera
"""

import argparse
import sys

import cv2
import torch
from ultralytics import YOLO


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Live webcam snack detection")
    parser.add_argument("--model", default="snacks_best.pt", help="path to trained weights")
    parser.add_argument("--camera", type=int, default=0, help="camera index (0 = built-in on most MacBooks)")
    parser.add_argument("--source", default=None, help="path to a video file, used instead of a live camera")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--device", default=None, help="override auto-detected device (mps/cuda/cpu)")
    args = parser.parse_args()

    device = args.device or pick_device()
    print(f"Using device: {device}")

    model = YOLO(args.model)

    capture_source = args.source if args.source is not None else args.camera
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        if args.source:
            print(f"Could not open video file '{args.source}'.")
        else:
            print(
                f"Could not open camera index {args.camera}. Try a different --camera value (0, 1, ...), "
                "and make sure your terminal app has camera access in "
                "System Settings > Privacy & Security > Camera."
            )
        sys.exit(1)

    print("Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame (camera disconnected or video ended).")
                break

            results = model.predict(source=frame, device=device, conf=args.conf, verbose=False)
            result = results[0]
            annotated = result.plot()

            for index, box in enumerate(result.boxes.xywh):
                x_center, y_center = box[0].item(), box[1].item()
                cls_id = int(result.boxes.cls[index].item())
                conf = result.boxes.conf[index].item()
                label = model.names[cls_id]
                print(f"  {label} conf={conf:.2f} -> center X: {x_center:.1f}, Y: {y_center:.1f}")

            cv2.imshow("Snack detector", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

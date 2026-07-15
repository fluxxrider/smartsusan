"""Full YOLOv8s training run on the synthetic snacks dataset.

Plain-script counterpart to 04_train_model.ipynb's smoke test — a 60-epoch
background run is easier to babysit via a log file than via
`jupyter nbconvert --execute`. Same prepare_dataset()/train_yolo_model()
logic as the notebook; only the epoch count, save_period, and script-vs-
notebook framing differ.
"""

import os
import random
import shutil
import time
import yaml
from ultralytics import YOLO

DATA_DIR = "snacks_yolo_dataset"
MODEL_NAME = "yolov8s.pt"
RUN_NAME = "snacks"
IMGSZ = 640
BATCH = 16
EPOCHS = 60
PATIENCE = 20
SAVE_PERIOD = 20  # checkpoint every 20 epochs, in addition to last.pt/best.pt every epoch


def prepare_dataset(data_dir):
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    yaml_path = os.path.join(data_dir, "data.yaml")
    train_img_dir = os.path.join(images_dir, "train")

    if not os.path.exists(train_img_dir):
        print("Train folder not found. Splitting data into train and val...")
        for folder in ["train", "val"]:
            os.makedirs(os.path.join(images_dir, folder), exist_ok=True)
            os.makedirs(os.path.join(labels_dir, folder), exist_ok=True)

        images = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and os.path.isfile(os.path.join(images_dir, f))
        ]
        random.seed(42)
        random.shuffle(images)
        split_idx = int(len(images) * 0.8)
        train_images, val_images = images[:split_idx], images[split_idx:]

        def move_files(file_list, dest_suffix):
            for img_name in file_list:
                shutil.move(os.path.join(images_dir, img_name), os.path.join(images_dir, dest_suffix, img_name))
                lbl_name = os.path.splitext(img_name)[0] + ".txt"
                lbl_src = os.path.join(labels_dir, lbl_name)
                if os.path.exists(lbl_src):
                    shutil.move(lbl_src, os.path.join(labels_dir, dest_suffix, lbl_name))

        move_files(train_images, "train")
        move_files(val_images, "val")
        print(f"Split {len(train_images)} train / {len(val_images)} val images.")
    else:
        print("Data already split into train/val. Skipping.")

    with open(yaml_path, "r") as f:
        data_config = yaml.safe_load(f)
    data_config["path"] = os.path.abspath(data_dir)
    data_config["train"] = "images/train"
    data_config["val"] = "images/val"
    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f, sort_keys=False)
    print("data.yaml paths updated.")

    return yaml_path


def train_yolo_model(data_dir, model_name, run_name, epochs, imgsz, batch, patience, save_period, device="cuda", script_dir=None):
    yaml_path = prepare_dataset(data_dir)

    print("Starting YOLO training...")
    model = YOLO(model_name)
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=run_name,
        device=device,
        patience=patience,
        save_period=save_period,
    )

    if script_dir and results and hasattr(results, "save_dir"):
        best_weights_path = os.path.join(results.save_dir, "weights", "best.pt")
        if os.path.exists(best_weights_path):
            destination_path = os.path.join(script_dir, f"{run_name}_best.pt")
            shutil.copy(best_weights_path, destination_path)
            print(f"Saved best weights to: {destination_path}")
        else:
            print("Training completed, but best.pt was not found.")

    return results


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    start = time.time()
    results = train_yolo_model(
        data_dir=DATA_DIR,
        model_name=MODEL_NAME,
        run_name=RUN_NAME,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        save_period=SAVE_PERIOD,
        device="cuda",
        script_dir=script_dir,
    )
    elapsed = time.time() - start
    print(f"\nFull training run finished in {elapsed / 60:.1f} minutes.")

import os
import random
import shutil
import yaml
from ultralytics import YOLO

def train_yolo_model(data_dir, model_name='yolov8n.pt', run_name='snacks', epochs=1, imgsz=640, device='mps', script_dir=None):
    """
    Splits dataset into train/val if needed, updates the data.yaml file, 
    kicks off YOLO training, and copies the best weights to the script's folder.
    """
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')
    yaml_path = os.path.join(data_dir, 'data.yaml')
    train_img_dir = os.path.join(images_dir, 'train')

    # --- 1. Auto-Split Data (Only if needed) ---
    if not os.path.exists(train_img_dir):
        print("Train folder not found. Splitting data into train and val...")

        # Create the necessary subdirectories
        for folder in ['train', 'val']:
            os.makedirs(os.path.join(images_dir, folder), exist_ok=True)
            os.makedirs(os.path.join(labels_dir, folder), exist_ok=True)

        # Get all images sitting in the main 'images' folder
        images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.PNG')) and os.path.isfile(os.path.join(images_dir, f))]

        # Shuffle and split (80% train, 20% val)
        random.seed(42)
        random.shuffle(images)
        split_idx = int(len(images) * 0.8)

        train_images = images[:split_idx]
        val_images = images[split_idx:]

        def move_files(file_list, dest_suffix):
            for img_name in file_list:
                # Move the image
                shutil.move(os.path.join(images_dir, img_name), os.path.join(images_dir, dest_suffix, img_name))
                # Move the matching label file if it exists
                lbl_name = os.path.splitext(img_name)[0] + '.txt'
                if os.path.exists(os.path.join(labels_dir, lbl_name)):
                    shutil.move(os.path.join(labels_dir, lbl_name), os.path.join(labels_dir, dest_suffix, lbl_name))

        move_files(train_images, 'train')
        move_files(val_images, 'val')
        print(f"✅ Successfully split {len(train_images)} training images and {len(val_images)} validation images.")
    else:
        print("✅ Data is already split into train/val folders. Skipping split step.")

       # --- 2. Auto-Fix data.yaml Paths ---
    print("Configuring data.yaml...")
    try:
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)

        # Force the correct pathing rules for YOLOv8
        # We set the absolute base path, and point the sub-folders relative to it
        data_config['path'] = os.path.abspath(data_dir)
        data_config['train'] = 'images/train'
        data_config['val'] = 'images/val'

        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, sort_keys=False)
        print("✅ Successfully updated data.yaml with perfect paths.")
    except Exception as e:
        print(f"⚠️ Could not update data.yaml automatically. Make sure the file exists! Error: {e}")

    # --- 3. Start Training ---
    print("🚀 Starting YOLO training...")
    model = YOLO(model_name)  # Loads the pretrained base model
    results = model.train(data=yaml_path, epochs=epochs, imgsz=imgsz, name=run_name, device=device)

    # --- 4. Copy the Trained Model to the Script Directory ---
    if script_dir and results and hasattr(results, 'save_dir'):
        try:
            # YOLO results.save_dir holds the path to the current run folder (e.g., 'runs/detect/snacks')
            best_weights_path = os.path.join(results.save_dir, 'weights', 'best.pt')
            
            if os.path.exists(best_weights_path):
                # We name the copied weight file based on the run name
                destination_path = os.path.join(script_dir, f"{run_name}_best.pt")
                shutil.copy(best_weights_path, destination_path)
                print(f"🎯 Saved best trained model directly to script folder: {destination_path}")
            else:
                print("⚠️ Training completed, but the expected 'best.pt' weights file could not be found.")
        except Exception as e:
            print(f"⚠️ Failed to copy best weights to script folder: {e}")

    return results


# --- 5. Execution Block ---
if __name__ == "__main__":
    # Get the directory where this script is saved
    SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
    
    DATASET_DIRECTORY = '/Users/amiosarker/dev/programming/2026/Tufts-ai-final/celcius_drink_kiwi_guava_transparent_yolo_dataset'
    
    # Run the training pipeline
    train_yolo_model(
        data_dir=DATASET_DIRECTORY,
        model_name='yolov8n.pt',
        run_name='snacks',
        epochs=1,
        imgsz=640,
        device='mps',  # Keeps using your Apple Silicon GPU
        script_dir=SCRIPT_DIRECTORY  # Passes the script directory to the function
    )
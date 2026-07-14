import os
import random
import yaml  # You might need to run: pip install pyyaml
from PIL import Image, UnidentifiedImageError

def generate_synthetic_yolo_dataset(fg_folder, bg_folder, output_folder=None, num_images=100):
    # Automatically name the output folder based on the foreground object folder
    if output_folder is None:
        fg_basename = os.path.basename(os.path.normpath(fg_folder))
        output_folder = f"{fg_basename}_yolo_dataset"

    # Setup standard YOLOv8 directory structure
    images_dir = os.path.join(output_folder, "images")
    labels_dir = os.path.join(output_folder, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Gather foregrounds (transparent objects) and backgrounds
    fg_files = [f for f in os.listdir(fg_folder) if f.lower().endswith('.png')]
    bg_files = [f for f in os.listdir(bg_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not fg_files:
        print(f"Error: No transparent PNGs found in '{fg_folder}'")
        return
    if not bg_files:
        print(f"Error: No background images found in '{bg_folder}'")
        return

    # Automatically map foreground filenames to unique YOLO class integer IDs
    unique_classes = sorted(list(set(os.path.splitext(f)[0].replace("_transparent", "") for f in fg_files)))
    class_to_id = {classname: idx for idx, classname in enumerate(unique_classes)}
    
    print(f"Detected classes: {list(class_to_id.keys())}")
    print(f"Generating {num_images} synthetic YOLO training images...")

    generated_count = 0
    attempts = 0
    max_attempts = num_images * 3  

    while generated_count < num_images and attempts < max_attempts:
        attempts += 1
        
        # 1. Select random background and foreground asset
        bg_name = random.choice(bg_files)
        fg_name = random.choice(fg_files)
        
        bg_path = os.path.join(bg_folder, bg_name)
        fg_path = os.path.join(fg_folder, fg_name)

        try:
            bg_img = Image.open(bg_path).convert("RGB")
            fg_img = Image.open(fg_path) 
        except (UnidentifiedImageError, IOError):
            continue

        # 2. Get the tightest bounding box of the actual object
        tight_bbox = fg_img.getbbox() 
        if not tight_bbox:
            continue 
        
        fg_cropped = fg_img.crop(tight_bbox)
        label_name = os.path.splitext(os.path.basename(fg_path))[0].replace("_transparent", "")
        class_id = class_to_id[label_name]

        # 3. Dynamically resize the object relative to the background width
        bg_w, bg_height = bg_img.size
        scale_factor = random.uniform(0.15, 0.35) 
        new_w = int(bg_w * scale_factor)
        
        aspect_ratio = fg_cropped.size[1] / fg_cropped.size[0]
        new_h = int(new_w * aspect_ratio)
        
        if new_w <= 0 or new_h <= 0:
            continue

        fg_resized = fg_cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 4. Pick a random placement coordinate
        max_x = bg_w - new_w
        max_y = bg_height - new_h
        
        if max_x <= 0 or max_y <= 0:
            continue
            
        paste_x = random.randint(0, max_x)
        paste_y = random.randint(0, max_y)

        # 5. Paste the foreground onto background
        bg_img.paste(fg_resized, (paste_x, paste_y), fg_resized)

        generated_count += 1

        # Define output filenames
        base_filename = f"synthetic_{generated_count}"
        img_filename = f"{base_filename}.jpg"
        txt_filename = f"{base_filename}.txt"

        # Save the image directly to the images folder
        bg_img.save(os.path.join(images_dir, img_filename), "JPEG")

        # 6. Calculate normalized YOLO bounding box values
        # YOLO format requires: class_id x_center y_center width height (all normalized 0.0 to 1.0)
        x_center = (paste_x + (new_w / 2.0)) / bg_w
        y_center = (paste_y + (new_h / 2.0)) / bg_height
        norm_w = new_w / bg_w
        norm_h = new_h / bg_height

        # 7. Write the YOLO txt annotation file
        txt_path = os.path.join(labels_dir, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        if generated_count % 10 == 0 or generated_count == num_images:
            print(f"  -> Generated {generated_count}/{num_images} files...")

    # 8. Automatically generate the data.yaml file required by YOLOv8
    yaml_data = {
        'path': os.path.abspath(output_folder),
        'train': 'images',
        'val': 'images',  # Using same folder for quick testing; split later if desired
        'names': {idx: name for name, idx in class_to_id.items()}
    }
    
    yaml_path = os.path.join(output_folder, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"\nDone! Your custom YOLOv8 dataset is ready in: /{output_folder}")
    print(f"Generated data.yaml mapping at: {yaml_path}")

if __name__ == "__main__":
    generate_synthetic_yolo_dataset(
        fg_folder="celcius_drink_kiwi_guava_transparent", 
        bg_folder="interior_spaces", 
        num_images=300
    )
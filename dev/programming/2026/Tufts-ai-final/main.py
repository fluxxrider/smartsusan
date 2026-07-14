import data_scraper as ds
import background_remover as br
import dataset_generator as dg
import image_rec_trainer as irt
from ultralytics import YOLO
import os


print("Welcome to the YOLOv8 Image Recognition Pipeline!")

forground = input("what snack u want:")

ammount = input("Enter the maximum number of images to download (default 100): ")
training_data_ammount = input("Enter the maximum number of images to generate for training (default 300): ")
ds.download_high_res_images_fallback(search_query=forground, max_images=int(ammount) if ammount.isdigit() else 100)
br.automate_local_remove_bg(input_folder=forground, output_folder=f"{forground}_transparent")
dg.generate_synthetic_yolo_dataset(
        fg_folder=f"{forground}_transparent", 
        bg_folder="background", 
        num_images=int(training_data_ammount) if training_data_ammount.isdigit() else 300
    )
data_dir=f"{forground}_transparent_yolo_dataset"
S = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
irt.train_yolo_model(
        data_dir=data_dir,
        model_name='yolov8n.pt',
        run_name=f"{forground}",
        epochs=100,
        imgsz=640,
        device='mps',
        script_dir = S  # Keeps using your Apple Silicon GPU
    )
# 1. Load your custom trained model
model = YOLO(f'{forground}_best.pt')

# 2. Run inference on a local image
# (We set show=False here since we just want the console text, but you can set it to True!)
results = model.predict(source='1', show=True, save=True, device='mps')
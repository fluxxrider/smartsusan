from ultralytics import YOLO

# 1. Load your custom trained model
model = YOLO('best.pt')

# 2. Run inference on a local image
# (We set show=False here since we just want the console text, but you can set it to True!)
results = model.predict(source='1', show=True, save=True, device='mps')

# 3. Extract and print the center coordinates
# Since we passed one image, we look at the first result: results[0]
result = results[0]

print(f"\nFound {len(result.boxes)} objects!")

# result.boxes.xywh gives a list of [x_center, y_center, width, height] for each box
for index, box in enumerate(result.boxes.xywh):
    # .item() converts the value from a PyTorch tensor to a standard Python number
    x_center = box[0].item() 
    y_center = box[1].item()
    
    print(f"Object {index + 1} -> Center X: {x_center:.2f}, Center Y: {y_center:.2f}")
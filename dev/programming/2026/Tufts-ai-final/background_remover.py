import os
from rembg import remove
from PIL import Image

def automate_local_remove_bg(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through every file in your input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            
            # Ensure the output is saved as a .png to keep transparency
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, f"{base_name}_transparent.png")

            print(f"Processing '{filename}'...")

            try:
                # Open the original image
                input_img = Image.open(input_path)
                
                # Pass it through the local AI model to remove the background
                output_img = remove(input_img)
                
                # Save the isolated object
                output_img.save(output_path)
                print(f"  -> Success! Saved to {output_path}")
                
            except Exception as e:
                print(f"  -> Failed to process {filename}: {e}")

    print("\nDone! All backgrounds removed locally with zero limits.")

if __name__ == "__main__":
    # Point these to your actual folders
    automate_local_remove_bg(input_folder="egg", output_folder="egg_transparent")
from PIL import Image
import os
from PIL import UnidentifiedImageError

input_dir = 'diatomic_dataset/images'
output_dir = 'diatomic_dataset/resized_images'

os.makedirs(output_dir, exist_ok=True)

allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

target_size = (256, 256)

problem_files = []

for filename in os.listdir(input_dir):
    file_path = os.path.join(input_dir, filename)
        
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        print(f"Skipping non-image file: {filename}")
        continue
        
    try:
        with Image.open(file_path) as img:
            # Resize image
            img = img.resize(target_size)
            
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
            img.save(output_path)
    except UnidentifiedImageError:
        print(f"Corrupted image: {filename}")
        problem_files.append(filename)
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        problem_files.append(filename)

print(f"\nProcessing complete. Problem files: {problem_files}")

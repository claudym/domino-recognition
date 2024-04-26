from PIL import Image
import os

# Paths relative to the project root
raw_dir = 'data/raw'
processed_dir = 'data/processed'

def process_image(file_path, output_size=(512, 512)):
    with Image.open(file_path) as img:
        img = img.resize(output_size, Image.LANCZOS)  # LANCZOS is recommended for downsampling
        processed_path = os.path.join(processed_dir, os.path.basename(file_path))
        img.save(processed_path)

def process_all_images():
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    for filename in os.listdir(raw_dir):
        file_path = os.path.join(raw_dir, filename)
        process_image(file_path)

if __name__ == '__main__':
    process_all_images()

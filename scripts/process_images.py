import argparse
from PIL import Image
import os

def process_image(file_path, output_dir, output_size=(224, 224)):
    """Process an image and save it to the specified directory."""
    with Image.open(file_path) as img:
        img = img.resize(output_size, Image.LANCZOS)  # LANCZOS is recommended for downsampling
        processed_path = os.path.join(output_dir, os.path.basename(file_path))
        img.save(processed_path)

def process_all_images(raw_dir, processed_dir):
    """Process all images in a directory and save them to another directory."""
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    for filename in os.listdir(raw_dir):
        file_path = os.path.join(raw_dir, filename)
        process_image(file_path, processed_dir)

def main():
    parser = argparse.ArgumentParser(description='Process images from one directory and save them to another.')
    parser.add_argument('raw_dir', type=str, help='Directory where the raw images are stored')
    parser.add_argument('processed_dir', type=str, help='Directory where the processed images should be saved')
    
    args = parser.parse_args()
    
    process_all_images(args.raw_dir, args.processed_dir)

if __name__ == '__main__':
    main()

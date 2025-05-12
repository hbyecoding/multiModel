import os
import requests
import tarfile
from pathlib import Path
import subprocess
import time

def setup_flickr30k_dataset():
    """Set up the Flickr30K dataset by creating necessary directories and downloading required files."""
    # Define base directory for dataset
    base_dir = Path('data/flickr30k')
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create images directory
    images_dir = base_dir / 'flickr30k_images'
    images_dir.mkdir(exist_ok=True)
    
    # Download and setup images
    if not any(images_dir.iterdir()):
        print("\nDownloading Flickr30K images...")
        try:
            # Download image parts
            image_parts = [
                "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part00",
                "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part01",
                "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part02"
            ]
            
            for i, url in enumerate(image_parts, 1):
                print(f"Downloading part {i}/3...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                part_file = base_dir / f"flickr30k_part{i-1:02d}"
                with open(part_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            # Combine parts and create zip file
            print("Combining parts...")
            with open(base_dir / 'flickr30k.zip', 'wb') as outfile:
                for i in range(3):
                    part_file = base_dir / f"flickr30k_part{i:02d}"
                    with open(part_file, 'rb') as infile:
                        outfile.write(infile.read())
                    part_file.unlink()
            
            # Extract images
            print("Extracting images...")
            subprocess.run(['unzip', '-q', str(base_dir / 'flickr30k.zip'), '-d', str(images_dir)], check=True)
            (base_dir / 'flickr30k.zip').unlink()
            
            print("Successfully downloaded and extracted images.")
            
        except Exception as e:
            print(f"Error downloading images: {e}")
            raise
    
    # Download annotations file
    annotations_url = "http://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k.tar.gz"
    annotations_file = base_dir / 'flickr30k.tar.gz'
    
    if not (base_dir / 'results_20130124.token').exists():
        print("Downloading Flickr30K annotations...")
        try:
            response = requests.get(annotations_url, stream=True)
            response.raise_for_status()
            
            with open(annotations_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the annotations file
            with tarfile.open(annotations_file) as tar:
                tar.extractall(path=base_dir)
            
            # Clean up the tar file
            annotations_file.unlink()
            
            print("Successfully downloaded and extracted annotations.")
        except Exception as e:
            print(f"Error downloading annotations: {e}")
            raise
    
    print("\nDataset setup complete!")
    print("Both images and annotations have been downloaded and extracted.")
    print(f"Dataset location: {base_dir}")
    print(f"Images location: {images_dir}")

if __name__ == "__main__":
    setup_flickr30k_dataset()
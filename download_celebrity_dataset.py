#!/usr/bin/env python3

import os
import shutil
import kagglehub
from pathlib import Path
import zipfile
import glob
import random
import sys

# Get the proper paths from the face recognition system
project_dir = Path(__file__).resolve().parent
standalone_face_dir = project_dir
if __file__.endswith('src/download_dataset.py'):
    standalone_face_dir = project_dir.parent
sys.path.insert(0, str(standalone_face_dir))

try:
    # Try to import from the face recognition system
    from standalone_face_recognition.src.base_config import RAW_DATA_DIR
    print(f"Using RAW_DATA_DIR: {RAW_DATA_DIR}")
except ImportError:
    # Fallback if we can't import
    RAW_DATA_DIR = project_dir / "data" / "raw"
    print(f"Using fallback RAW_DATA_DIR: {RAW_DATA_DIR}")

# Simple dataset definitions
DATASETS = {
    "celebrity_faces": {
        "kaggle_id": "vishesh1412/celebrity-face-image-dataset", 
        "description": "18 celebrities with ~100 images each",
    },
    "lfw": {
        "kaggle_id": "atulanandjha/lfwpeople",
        "description": "5749 people with varying numbers of images from LFW dataset",
    },
    "face_recognition": {
        "kaggle_id": "cybersimar08/face-recognition-dataset",
        "description": "Face recognition dataset with multiple subjects",
    }
}

def clean_person_name(name):
    """Clean up person name from dataset-specific formats."""
    # Handle "Celebrity Faces Dataset_Angelina Jolie" format
    if "Celeberity Faces Dataset_" in name or "Celebrity Faces Dataset_" in name:
        return name.split("_", 1)[1].strip()
    
    # Handle other common prefixes
    prefixes = ["lfw_", "face_", "person_", "subj_"]
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]
    
    return name

def scan_for_person_directories(root_dir, dataset_id=None):
    """
    Scan for person directories containing images in dataset.
    
    Some datasets (like LFW) have multiple levels of directories or 
    special folders like "Detected Faces"/"Faces" that need to be searched.
    """
    person_dirs = []
    
    # Look for top-level person directories first
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            # Check if this directory contains images directly
            images = glob.glob(os.path.join(item_path, "*.jpg")) + glob.glob(os.path.join(item_path, "*.png"))
            if images:
                # For Celebrity Faces Dataset, clean the name
                if dataset_id == "celebrity_faces" and ("Celeberity Faces Dataset_" in item or "Celebrity Faces Dataset_" in item):
                    person_name = clean_person_name(item)
                    person_dirs.append((person_name, item_path))
                else:
                    person_dirs.append((item, item_path))
            else:
                # No images directly in this directory, check its subdirectories
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isdir(subitem_path):
                        # Check if this subdirectory contains images
                        subimages = glob.glob(os.path.join(subitem_path, "*.jpg")) + glob.glob(os.path.join(subitem_path, "*.png"))
                        if subimages:
                            # For LFW-type datasets, use both the parent and child directory names
                            # (e.g., "Detected Faces/George_W_Bush" -> "George_W_Bush")
                            if item in ["Detected Faces", "Faces", "faces", "detected_faces", "lfw"]:
                                person_dirs.append((subitem, subitem_path))
                            else:
                                person_dirs.append((f"{item}_{subitem}", subitem_path))
    
    return person_dirs

def extract_images(dataset_path, target_dir, max_celebrities=20, images_per_celebrity=20, dataset_id=None):
    """Extract images from the dataset to the target directory."""
    # Special handling for celebrity_faces dataset
    if dataset_id == "celebrity_faces":
        # Look for "Celebrity Faces Dataset" directory
        celeb_dir = os.path.join(dataset_path, "Celebrity Faces Dataset")
        if os.path.exists(celeb_dir) and os.path.isdir(celeb_dir):
            dataset_path = celeb_dir
    
    # First, try to find person directories in the dataset
    person_dirs = scan_for_person_directories(dataset_path, dataset_id)
    
    # Process directories if found
    if person_dirs:
        # Limit to max_celebrities
        if max_celebrities and len(person_dirs) > max_celebrities:
            random.shuffle(person_dirs)
            person_dirs = person_dirs[:max_celebrities]
        
        total_images = 0
        # Process each person directory
        for person_name, person_dir in person_dirs:
            # Clean up person name
            person_name = clean_person_name(person_name)
            
            # Create directory for this person
            person_target_dir = target_dir / person_name
            person_target_dir.mkdir(exist_ok=True)
            
            # Get and limit images
            images = glob.glob(os.path.join(person_dir, "*.jpg")) + glob.glob(os.path.join(person_dir, "*.png"))
            if images_per_celebrity and len(images) > images_per_celebrity:
                random.shuffle(images)
                images = images[:images_per_celebrity]
            
            # Copy images
            for i, img_path in enumerate(images):
                target_path = person_target_dir / f"{person_name}_{i:03d}{os.path.splitext(img_path)[1]}"
                shutil.copy2(img_path, target_path)
                total_images += 1
        
        return len(person_dirs), total_images
    
    # If no directories with images found, look for images directly in the dataset
    images = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(root, file))
    
    if not images:
        return 0, 0
    
    # Try to extract person names from filenames
    person_images = {}
    for img_path in images:
        filename = os.path.basename(img_path)
        
        # For celebrity faces dataset, handle special format
        if dataset_id == "celebrity_faces" and ("Celeberity Faces Dataset_" in filename or "Celebrity Faces Dataset_" in filename):
            # Format: "Celeberity Faces Dataset_Angelina Jolie_1.jpg"
            parts = os.path.splitext(filename)[0].split("_")
            if len(parts) >= 3:  # Need at least 3 parts for this format
                person_name = "_".join(parts[1:-1])  # Join the middle parts
                person_name = clean_person_name(person_name)
            else:
                person_name = f"person_{len(person_images) + 1}"
        else:
            # Standard processing
            name_parts = os.path.splitext(filename)[0].split('_')
            if len(name_parts) > 1:
                person_name = name_parts[0]
            else:
                name_parts = os.path.splitext(filename)[0].split('-')
                if len(name_parts) > 1:
                    person_name = name_parts[0]
                else:
                    # Use parent directory name as a fallback
                    parent_dir = os.path.basename(os.path.dirname(img_path))
                    if parent_dir and parent_dir not in ["dataset", "images", "imgs", "data"]:
                        person_name = parent_dir
                    else:
                        person_name = f"person_{len(person_images) + 1}"
        
        # Clean up the person name
        person_name = clean_person_name(person_name)
        
        if person_name not in person_images:
            person_images[person_name] = []
        
        person_images[person_name].append(img_path)
    
    # Limit to max_celebrities
    person_names = list(person_images.keys())
    if max_celebrities and len(person_names) > max_celebrities:
        person_names = person_names[:max_celebrities]
    
    # Process each person's images
    total_images = 0
    for person_name in person_names:
        person_target_dir = target_dir / person_name
        person_target_dir.mkdir(exist_ok=True)
        
        # Limit images per person
        person_imgs = person_images[person_name]
        if images_per_celebrity and len(person_imgs) > images_per_celebrity:
            random.shuffle(person_imgs)
            person_imgs = person_imgs[:images_per_celebrity]
        
        # Copy images
        for i, img_path in enumerate(person_imgs):
            target_path = person_target_dir / f"{person_name}_{i:03d}{os.path.splitext(img_path)[1]}"
            shutil.copy2(img_path, target_path)
            total_images += 1
    
    return len(person_names), total_images

def handle_lfw_dataset(dataset_path, target_dir, max_celebrities=20, images_per_celebrity=20):
    """Special handling for the LFW dataset which has a unique structure."""
    # The LFW dataset has a specific structure: each person has their own directory
    # First, look for the lfw directory structure
    people_dirs = []
    
    # Check all possible locations for LFW directories
    for root, dirs, _ in os.walk(dataset_path):
        for d in dirs:
            # Look for directories that might be person folders (typical LFW structure)
            person_dir = os.path.join(root, d)
            # Only consider if it has enough images
            images = glob.glob(os.path.join(person_dir, "*.jpg"))
            if len(images) >= 4:  # Often celebrities have at least 4 images
                people_dirs.append((d, person_dir))
    
    if not people_dirs:
        print("Could not find LFW-formatted directories with person images")
        return 0, 0
    
    # Limit to max_celebrities with most images
    people_dirs.sort(key=lambda x: len(glob.glob(os.path.join(x[1], "*.jpg"))), reverse=True)
    if max_celebrities and len(people_dirs) > max_celebrities:
        people_dirs = people_dirs[:max_celebrities]
    
    total_images = 0
    # Process each person directory
    for person_name, person_dir in people_dirs:
        person_name = clean_person_name(person_name)
        
        # Create directory for this person
        person_target_dir = target_dir / person_name
        person_target_dir.mkdir(exist_ok=True)
        
        # Get all images for this person
        images = glob.glob(os.path.join(person_dir, "*.jpg"))
        if images_per_celebrity and len(images) > images_per_celebrity:
            random.shuffle(images)
            images = images[:images_per_celebrity]
        
        # Copy images
        for i, img_path in enumerate(images):
            target_path = person_target_dir / f"{person_name}_{i:03d}{os.path.splitext(img_path)[1]}"
            shutil.copy2(img_path, target_path)
            total_images += 1
    
    return len(people_dirs), total_images

def download_dataset(dataset_id, max_celebrities=20, images_per_celebrity=20):
    """Download a specific dataset and organize it."""
    if dataset_id not in DATASETS:
        print(f"Unknown dataset: {dataset_id}")
        return False
    
    # Create target directory
    target_dir = RAW_DATA_DIR / dataset_id
    
    # Skip if already downloaded
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"Dataset {dataset_id} already exists at {target_dir}")
        return True
    
    # Make sure directory exists
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    kaggle_id = DATASETS[dataset_id]["kaggle_id"]
    print(f"Downloading {dataset_id} ({DATASETS[dataset_id]['description']})...")
    print(f"Kaggle ID: {kaggle_id}")
    
    try:
        # Download from Kaggle
        dataset_path = kagglehub.dataset_download(kaggle_id)
        print(f"Downloaded to: {dataset_path}")
        
        # Extract dataset to temp directory
        temp_dir = project_dir / f"temp_{dataset_id}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()
        
        # Look for zip files to extract
        zip_files = glob.glob(os.path.join(dataset_path, "*.zip"))
        if zip_files:
            for zip_file in zip_files:
                print(f"Extracting {zip_file}...")
                try:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                except zipfile.BadZipFile:
                    print(f"Warning: Invalid zip file {zip_file}, skipping...")
                    continue
        
        # Copy all files - needed for LFW which may not have zip files
        for item in os.listdir(dataset_path):
            src = Path(dataset_path) / item
            dst = temp_dir / item
            if src.is_dir():
                if not (dst.exists()):  # Avoid copying if already extracted from zip
                    shutil.copytree(src, dst)
            elif not dst.exists():  # Avoid copying if already extracted
                shutil.copy2(src, dst)
        
        # Process based on dataset type
        if dataset_id == "lfw":
            print(f"Processing LFW dataset with special handling...")
            num_persons, num_images = handle_lfw_dataset(
                temp_dir, target_dir, 
                max_celebrities=max_celebrities,
                images_per_celebrity=images_per_celebrity
            )
        else:
            # Extract and organize images
            print(f"Organizing images for {dataset_id}...")
            num_persons, num_images = extract_images(
                temp_dir, target_dir, 
                max_celebrities=max_celebrities,
                images_per_celebrity=images_per_celebrity,
                dataset_id=dataset_id
            )
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        if num_images > 0:
            print(f"Successfully organized {num_images} images for {num_persons} persons in {target_dir}")
            
            # Create info.txt file with dataset information
            with open(target_dir / "info.txt", "w") as f:
                f.write(f"Dataset: {dataset_id}\n")
                f.write(f"Description: {DATASETS[dataset_id]['description']}\n")
                f.write(f"Kaggle ID: {kaggle_id}\n")
                f.write(f"Number of persons: {num_persons}\n")
                f.write(f"Number of images: {num_images}\n")
            
            return True
        else:
            print(f"No images found in the dataset. Removing {target_dir}")
            shutil.rmtree(target_dir)
            return False
            
    except Exception as e:
        print(f"Error downloading/extracting dataset {dataset_id}: {e}")
        # Clean up if there was an error
        if target_dir.exists():
            shutil.rmtree(target_dir)
        return False

def download_all_datasets(max_celebrities=20, images_per_celebrity=20):
    """Download all defined datasets."""
    success = True
    for dataset_id in DATASETS:
        print(f"\n{'='*60}")
        result = download_dataset(dataset_id, max_celebrities, images_per_celebrity)
        print(f"{'='*60}")
        success = success and result
    
    return success

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download face recognition datasets')
    parser.add_argument('--dataset', type=str, choices=list(DATASETS.keys()), 
                        help='Specific dataset to download')
    parser.add_argument('--max-celebrities', type=int, default=20, 
                        help='Maximum number of celebrities per dataset')
    parser.add_argument('--images-per-celebrity', type=int, default=20,
                        help='Maximum images per celebrity')
    
    args = parser.parse_args()
    
    # Create raw data directory
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.dataset:
        download_dataset(args.dataset, args.max_celebrities, args.images_per_celebrity)
    else:
        print("Downloading all datasets...")
        download_all_datasets(args.max_celebrities, args.images_per_celebrity)
        print("\nDatasets downloaded to subdirectories in:", RAW_DATA_DIR)
        
    print("\nUse 'python run.py interactive' to process and train models with these datasets.") 
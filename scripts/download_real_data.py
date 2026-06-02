"""
MedVision Real Dataset Download Script
Downloads and structures:
  1. Chest X-Ray Images (Pneumonia) from Hugging Face Hub (parquet format)
  2. Montgomery County CXR (Lung Segmentation) from NLM
"""

import os
import sys
import zipfile
import shutil
import io
import urllib.request
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download, list_repo_files

# Adjust path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def download_file(url: str, output_path: str, desc: str = ""):
    """Download file with progress bar."""
    print(f"\nDownloading: {desc or url}")
    print(f"Target: {output_path}")
    
    # Simple hook for progress
    pbar = None
    
    def progress_hook(block_count, block_size, total_size):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=desc)
        downloaded = block_count * block_size
        pbar.update(min(block_size, total_size - downloaded))
        
    try:
        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
        if pbar:
            pbar.close()
        print("\nDownload complete.")
        return True
    except Exception as e:
        if pbar:
            pbar.close()
        print(f"\n[ERROR] Download failed: {e}")
        return False


def setup_real_classification_data():
    """
    Downloads the Chest X-Ray Pneumonia dataset from Hugging Face.
    Extracts images from parquet files and saves them to:
        data/chest_xray/{train|val|test}/{NORMAL|PNEUMONIA}/
    """
    print("\n" + "=" * 60)
    print("  Setting up Chest X-Ray Classification Dataset (Hugging Face)")
    print("=" * 60)
    
    xray_dir = Path(config.CHEST_XRAY_DIR)
    
    # Check if we already have real data (we'll look for a larger count than synthetic 50)
    train_normal_dir = xray_dir / "train" / "NORMAL"
    if train_normal_dir.exists() and len(os.listdir(train_normal_dir)) > 100:
        print("  Real Chest X-Ray dataset already exists. Skipping download.")
        return
        
    # Clear existing synthetic classification data if it exists
    if xray_dir.exists():
        print("  Removing synthetic chest X-ray data...")
        shutil.rmtree(xray_dir, ignore_errors=True)
    
    # Create target directories
    for split in ["train", "val", "test"]:
        for cls in ["NORMAL", "PNEUMONIA"]:
            os.makedirs(xray_dir / split / cls, exist_ok=True)
            
    print("  Downloading parquet files from Hugging Face hub...")
    repo_id = "hf-vision/chest-xray-pneumonia"
    
    try:
        files = list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        print(f"  [ERROR] Failed to list HF repo files: {e}")
        return
        
    parquet_files = [f for f in files if f.endswith(".parquet")]
    print(f"  Found {len(parquet_files)} parquet files to download.")
    
    # Download and process files
    for p_file in sorted(parquet_files):
        print(f"  Processing {p_file}...")
        
        # Determine split
        if "train" in p_file:
            split = "train"
        elif "validation" in p_file:
            split = "val"
        elif "test" in p_file:
            split = "test"
        else:
            continue
            
        # Download file
        local_path = hf_hub_download(repo_id, p_file, repo_type="dataset")
        
        # Read parquet
        print(f"  Reading images from {p_file}...")
        df = pd.read_parquet(local_path)
        
        # Iterate rows
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split}"):
            label = "PNEUMONIA" if row["label"] == 1 else "NORMAL"
            img_dict = row["image"]
            
            # Load and save image
            try:
                img_bytes = img_dict["bytes"]
                img = Image.open(io.BytesIO(img_bytes))
                
                # Save path
                out_path = xray_dir / split / label / f"{split}_{idx:04d}_{os.path.basename(p_file).replace('.parquet', '')}.jpeg"
                img.save(out_path)
            except Exception as e:
                print(f"\n  [WARN] Failed to process image {idx} in {p_file}: {e}")
                
    print("\n  Chest X-Ray classification dataset successfully set up!")
    
    # Print counts
    for split in ["train", "val", "test"]:
        n_normal = len(os.listdir(xray_dir / split / "NORMAL"))
        n_pneu = len(os.listdir(xray_dir / split / "PNEUMONIA"))
        print(f"    - {split}: NORMAL={n_normal}, PNEUMONIA={n_pneu} (Total: {n_normal + n_pneu})")


def setup_real_segmentation_data():
    """
    Downloads the Montgomery County lung segmentation dataset from NLM.
    Extracts original X-rays and merges left/right masks, saving them to:
        data/segmentation/images/
        data/segmentation/masks/
    """
    print("\n" + "=" * 60)
    print("  Setting up Lung Segmentation Dataset (Montgomery County)")
    print("=" * 60)
    
    seg_dir = Path(config.SEGMENTATION_DIR)
    images_dir = seg_dir / "images"
    masks_dir = seg_dir / "masks"
    
    # Check if we already have real data
    if images_dir.exists() and len(os.listdir(images_dir)) > 100:
        print("  Real segmentation dataset already exists. Skipping.")
        return
        
    # Clear existing synthetic segmentation data if it exists
    if seg_dir.exists():
        print("  Removing synthetic segmentation data...")
        shutil.rmtree(seg_dir, ignore_errors=True)
        
    # Create target directories
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # NLM Montgomery ZIP URL
    url = "https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip"
    zip_path = seg_dir / "NLM-MontgomeryCXRSet.zip"
    
    # Download zip file
    success = download_file(url, str(zip_path), desc="Montgomery CXR ZIP")
    if not success:
        print("  [ERROR] Failed to download Montgomery dataset.")
        return
        
    # Extract ZIP file
    print("  Extracting ZIP file...")
    temp_extract_dir = seg_dir / "temp_extract"
    os.makedirs(temp_extract_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
    except Exception as e:
        print(f"  [ERROR] Extraction failed: {e}")
        # Clean up
        shutil.rmtree(temp_extract_dir, ignore_errors=True)
        if zip_path.exists():
            os.remove(zip_path)
        return
        
    # Locate folders (sometimes inside nested folders like NLM-MontgomeryCXRSet/MontgomerySet/)
    set_root = None
    for root, dirs, files in os.walk(temp_extract_dir):
        if "CXR_png" in dirs and "ManualMask" in dirs:
            set_root = Path(root)
            break
            
    if not set_root:
        print("  [ERROR] Could not locate CXR_png and ManualMask folders in extracted files.")
        # Clean up
        shutil.rmtree(temp_extract_dir, ignore_errors=True)
        os.remove(zip_path)
        return
        
    cxr_dir = set_root / "CXR_png"
    left_mask_dir = set_root / "ManualMask" / "leftMask"
    right_mask_dir = set_root / "ManualMask" / "rightMask"
    
    xray_images = sorted([f for f in os.listdir(cxr_dir) if f.endswith(".png")])
    print(f"  Processing {len(xray_images)} images and merging masks...")
    
    for img_name in tqdm(xray_images, desc="Merging Masks"):
        img_src = cxr_dir / img_name
        left_mask_src = left_mask_dir / img_name
        right_mask_src = right_mask_dir / img_name
        
        if not left_mask_src.exists() or not right_mask_src.exists():
            print(f"  [WARN] Missing masks for image {img_name}, skipping.")
            continue
            
        # Move image
        shutil.copy2(img_src, images_dir / img_name)
        
        # Merge masks (Left + Right lungs)
        try:
            left_mask = np.array(Image.open(left_mask_src).convert("L"))
            right_mask = np.array(Image.open(right_mask_src).convert("L"))
            
            # Combine via maximum
            combined_mask = np.maximum(left_mask, right_mask)
            
            # Save combined mask
            Image.fromarray(combined_mask).save(masks_dir / img_name)
        except Exception as e:
            print(f"  [ERROR] Error merging masks for {img_name}: {e}")
            
    # Clean up temporary folders and zip
    print("  Cleaning up temporary files...")
    shutil.rmtree(temp_extract_dir, ignore_errors=True)
    if zip_path.exists():
        os.remove(zip_path)
        
    print(f"  Lung segmentation dataset successfully set up! Total images: {len(os.listdir(images_dir))}")


def main():
    print("=" * 60)
    print("  MedVision — Real Dataset Ingestion & Setup")
    print("=" * 60)
    
    setup_real_classification_data()
    setup_real_segmentation_data()
    
    print("\n" + "=" * 60)
    print("  Real dataset ingestion completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

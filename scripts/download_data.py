"""
MedVision Data Download Script
Downloads public medical imaging datasets for training and evaluation.

Datasets:
    1. Chest X-Ray Pneumonia (Kaggle/Mendeley) — Classification
    2. Montgomery County CXR — Lung Segmentation Masks
    3. Sample DICOM files — DICOM handling demos
"""

import os
import sys
import zipfile
import shutil
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def download_file(url: str, output_path: str, desc: str = ""):
    """Download a file with progress display."""
    print(f"  Downloading: {desc or url}")
    print(f"  Target: {output_path}")
    
    def progress_hook(block_count, block_size, total_size):
        downloaded = block_count * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)
    
    try:
        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
        print()  # Newline after progress
        return True
    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        return False


def setup_chest_xray_data():
    """
    Set up chest X-ray classification dataset.
    
    Uses the Chest X-Ray Pneumonia dataset structure:
        chest_xray/
            train/
                NORMAL/
                PNEUMONIA/
            val/
                NORMAL/
                PNEUMONIA/
            test/
                NORMAL/
                PNEUMONIA/
    
    If not available for download, creates a synthetic mini-dataset for testing.
    """
    print("\n" + "=" * 60)
    print("  Setting up Chest X-Ray Classification Dataset")
    print("=" * 60)
    
    data_dir = config.CHEST_XRAY_DIR
    
    # Check if already exists
    train_dir = os.path.join(data_dir, "train")
    if os.path.exists(train_dir) and len(os.listdir(train_dir)) > 0:
        print("  Dataset already exists. Skipping download.")
        return
    
    print("\n  NOTE: The full Chest X-Ray dataset requires manual download from Kaggle.")
    print("  URL: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("  After downloading, extract to:", data_dir)
    print("\n  Creating synthetic mini-dataset for testing...")
    
    # Create synthetic data for immediate testing
    _create_synthetic_classification_data(data_dir)


def setup_segmentation_data():
    """
    Set up lung segmentation dataset.
    
    Uses Montgomery County CXR dataset or creates synthetic data.
    
    Structure:
        segmentation/
            images/
                img001.png
            masks/
                img001.png (binary lung masks)
    """
    print("\n" + "=" * 60)
    print("  Setting up Lung Segmentation Dataset")
    print("=" * 60)
    
    data_dir = config.SEGMENTATION_DIR
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")
    
    if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0:
        print("  Dataset already exists. Skipping.")
        return
    
    print("\n  NOTE: Montgomery County CXR dataset available at:")
    print("  https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/")
    print("  Download and extract CXR images and masks to:", data_dir)
    print("\n  Creating synthetic mini-dataset for testing...")
    
    _create_synthetic_segmentation_data(data_dir)


def setup_dicom_samples():
    """
    Create sample DICOM files for testing DICOM handling.
    """
    print("\n" + "=" * 60)
    print("  Setting up DICOM Samples")
    print("=" * 60)
    
    dicom_dir = config.DICOM_DIR
    
    # Check if samples already exist
    existing = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    if existing:
        print(f"  {len(existing)} DICOM samples already exist. Skipping.")
        return
    
    from src.dicom_handler import create_sample_dicom, PYDICOM_AVAILABLE
    
    if not PYDICOM_AVAILABLE:
        print("  [WARN] pydicom not installed. Skipping DICOM sample creation.")
        return
    
    # Create sample DICOMs with various patient info (for de-identification testing)
    samples = [
        {
            "filename": "sample_chest_pa.dcm",
            "patient_name": "Smith^John",
            "modality": "CR",
        },
        {
            "filename": "sample_chest_lateral.dcm",
            "patient_name": "Doe^Jane",
            "modality": "CR",
        },
        {
            "filename": "sample_ct_chest.dcm",
            "patient_name": "Johnson^Robert",
            "modality": "CT",
        },
    ]
    
    for sample in samples:
        output_path = os.path.join(dicom_dir, sample["filename"])
        create_sample_dicom(
            output_path=output_path,
            patient_name=sample["patient_name"],
            modality=sample["modality"],
        )
    
    print(f"  Created {len(samples)} sample DICOM files")


def setup_sample_reports():
    """
    Create sample medical report images for OCR testing.
    """
    print("\n" + "=" * 60)
    print("  Setting up Sample Medical Reports")
    print("=" * 60)
    
    reports_dir = config.REPORTS_DIR
    
    existing = [f for f in os.listdir(reports_dir) if f.endswith('.png')]
    if existing:
        print(f"  {len(existing)} sample reports already exist. Skipping.")
        return
    
    _create_sample_medical_report(reports_dir)


def _create_synthetic_classification_data(data_dir: str, n_per_class: int = 50):
    """
    Create synthetic chest X-ray-like images for testing.
    Images are grayscale with simulated lung fields.
    """
    from PIL import Image as PILImage
    import numpy as np
    
    np.random.seed(42)
    
    for split in ["train", "val", "test"]:
        n = n_per_class if split == "train" else n_per_class // 5
        
        for class_name in config.CLASS_NAMES:
            class_dir = os.path.join(data_dir, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for i in range(n):
                # Generate synthetic X-ray-like image
                img = _generate_synthetic_xray(
                    size=256,
                    has_pathology=(class_name == "PNEUMONIA"),
                )
                
                pil_img = PILImage.fromarray(img)
                pil_img.save(os.path.join(class_dir, f"{class_name.lower()}_{split}_{i:04d}.png"))
    
    total = sum(
        len(os.listdir(os.path.join(data_dir, s, c)))
        for s in ["train", "val", "test"]
        for c in config.CLASS_NAMES
        if os.path.exists(os.path.join(data_dir, s, c))
    )
    print(f"  Created {total} synthetic X-ray images")


def _create_synthetic_segmentation_data(data_dir: str, n_samples: int = 40):
    """Create synthetic image-mask pairs for segmentation testing."""
    from PIL import Image as PILImage
    import numpy as np
    
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    np.random.seed(42)
    size = 256
    
    for i in range(n_samples):
        # Generate X-ray image
        img = _generate_synthetic_xray(size=size, has_pathology=False)
        
        # Generate corresponding lung mask
        mask = _generate_lung_mask(size=size)
        
        PILImage.fromarray(img).save(os.path.join(images_dir, f"img_{i:04d}.png"))
        PILImage.fromarray(mask).save(os.path.join(masks_dir, f"img_{i:04d}.png"))
    
    print(f"  Created {n_samples} image-mask pairs")


def _generate_synthetic_xray(size: int = 256, has_pathology: bool = False) -> np.ndarray:
    """Generate a synthetic chest X-ray-like image."""
    import numpy as np
    
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    
    # Background gradient (darker at top = air, brighter at bottom = diaphragm)
    background = 0.3 + 0.4 * y + np.random.normal(0, 0.02, (size, size))
    
    # Left lung field (dark ellipse)
    lung_left = np.exp(-((x - 0.35)**2 / 0.015 + (y - 0.45)**2 / 0.04))
    # Right lung field
    lung_right = np.exp(-((x - 0.65)**2 / 0.015 + (y - 0.45)**2 / 0.04))
    
    # Mediastinum (bright vertical band)
    mediastinum = 0.3 * np.exp(-((x - 0.5)**2 / 0.003))
    
    # Combine
    img = background - 0.4 * lung_left - 0.4 * lung_right + mediastinum
    
    # Add pathology (bright patches = consolidation/pneumonia)
    if has_pathology:
        # Random opacification in lung fields
        cx = np.random.choice([0.3, 0.35, 0.6, 0.65])
        cy = np.random.uniform(0.35, 0.55)
        patch = 0.3 * np.exp(-((x - cx)**2 / 0.005 + (y - cy)**2 / 0.008))
        img += patch
    
    # Add noise
    img += np.random.normal(0, 0.03, (size, size))
    
    # Normalize to uint8
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    
    return img


def _generate_lung_mask(size: int = 256) -> np.ndarray:
    """Generate a synthetic lung segmentation mask."""
    import numpy as np
    
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    
    # Left lung
    left = ((x - 0.35)**2 / 0.018 + (y - 0.45)**2 / 0.05) < 1.0
    # Right lung
    right = ((x - 0.65)**2 / 0.018 + (y - 0.45)**2 / 0.05) < 1.0
    
    mask = (left | right).astype(np.uint8) * 255
    return mask


def _create_sample_medical_report(reports_dir: str):
    """Create a sample medical report image for OCR testing."""
    from PIL import Image as PILImage, ImageDraw, ImageFont
    
    # Create white background
    img = PILImage.new('RGB', (800, 600), 'white')
    draw = ImageDraw.Draw(img)
    
    # Use default font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_bold = ImageFont.truetype("arialbd.ttf", 18)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_bold = font
    
    # Draw medical report content
    report_text = [
        ("RADIOLOGY REPORT", 30, 20, font_bold),
        ("", 0, 0, font),
        ("Patient Name: John Smith", 30, 60, font),
        ("Patient ID: MRN123456", 30, 85, font),
        ("Date of Birth: 03/15/1985", 30, 110, font),
        ("Date of Exam: 01/15/2024", 30, 135, font),
        ("Referring Physician: Dr. Sarah Johnson", 30, 160, font),
        ("Phone: (555) 123-4567", 30, 185, font),
        ("Email: john.smith@email.com", 30, 210, font),
        ("", 0, 0, font),
        ("EXAMINATION: Chest PA and Lateral", 30, 250, font_bold),
        ("", 0, 0, font),
        ("CLINICAL INDICATION:", 30, 280, font_bold),
        ("45 years old male with persistent cough.", 30, 305, font),
        ("", 0, 0, font),
        ("FINDINGS:", 30, 340, font_bold),
        ("The lungs are clear bilaterally. No focal", 30, 365, font),
        ("consolidation, pleural effusion, or pneumothorax.", 30, 390, font),
        ("The cardiac silhouette is normal in size.", 30, 415, font),
        ("The mediastinal contours are unremarkable.", 30, 440, font),
        ("", 0, 0, font),
        ("IMPRESSION:", 30, 475, font_bold),
        ("Normal chest radiograph. No acute findings.", 30, 500, font),
        ("", 0, 0, font),
        ("Radiologist: Dr. Michael Brown", 30, 540, font),
        ("Institution: Sample Medical Center", 30, 565, font),
    ]
    
    for text, x, y, f in report_text:
        if text:
            draw.text((x, y), text, fill='black', font=f)
    
    output_path = os.path.join(reports_dir, "sample_report.png")
    img.save(output_path)
    print(f"  Created sample medical report: {output_path}")


def main():
    """Download and set up all datasets."""
    print("=" * 60)
    print("  MedVision — Data Setup")
    print("=" * 60)
    
    setup_chest_xray_data()
    setup_segmentation_data()
    setup_dicom_samples()
    setup_sample_reports()
    
    print("\n" + "=" * 60)
    print("  Data setup complete!")
    print("=" * 60)
    print(f"  Classification data: {config.CHEST_XRAY_DIR}")
    print(f"  Segmentation data:   {config.SEGMENTATION_DIR}")
    print(f"  DICOM samples:       {config.DICOM_DIR}")
    print(f"  Sample reports:      {config.REPORTS_DIR}")
    print("\n  For best results, download real datasets:")
    print("  - Classification: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("  - Segmentation:   https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/")
    print("=" * 60)


if __name__ == "__main__":
    main()

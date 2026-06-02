"""
MedVision Configuration
Central configuration for all hyperparameters, paths, and device settings.
Medical imaging toolkit for classification, segmentation, DICOM handling, and de-identification.
"""

import os
import torch

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Classification Model (Chest X-Ray)
# ──────────────────────────────────────────────
CLASSIFIER_BACKBONE = "resnet18"
CLASSIFIER_PRETRAINED = True
NUM_CLASSES = 2                     # NORMAL, PNEUMONIA (extendable to 14 for ChestX-ray14)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
FREEZE_LAYERS = ["layer1"]          # Freeze earliest layers for transfer learning

# ──────────────────────────────────────────────
# Segmentation Model (U-Net)
# ──────────────────────────────────────────────
UNET_IN_CHANNELS = 1               # Grayscale X-ray input
UNET_OUT_CHANNELS = 1              # Binary lung mask
UNET_FEATURES = [64, 128, 256, 512]  # Encoder channel progression
UNET_BILINEAR = True               # Use bilinear upsampling (faster, less memory)

# ──────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────
BATCH_SIZE = 32
NUM_EPOCHS_CLASSIFIER = 15         # Classification epochs
NUM_EPOCHS_SEGMENTER = 25          # Segmentation epochs
LR_BACKBONE = 1e-4                 # Fine-tuning learning rate
LR_HEAD = 1e-3                     # Head learning rate
WEIGHT_DECAY = 1e-4
PATIENCE = 5                       # Early stopping patience

# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
IMAGE_SIZE = 224                    # Classification input size
SEG_IMAGE_SIZE = 256                # Segmentation input size (must be divisible by 16)
NUM_WORKERS = 2                    # DataLoader workers (Windows-safe)
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# ──────────────────────────────────────────────
# Medical Image Preprocessing
# ──────────────────────────────────────────────
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8
# CT Windowing presets (center, width) in HU
LUNG_WINDOW = (-600, 1500)
BONE_WINDOW = (400, 1800)
SOFT_TISSUE_WINDOW = (50, 400)
BRAIN_WINDOW = (40, 80)

# ──────────────────────────────────────────────
# De-identification
# ──────────────────────────────────────────────
PHI_TAGS = [
    "PatientName", "PatientID", "PatientBirthDate",
    "PatientSex", "PatientAge", "PatientAddress",
    "InstitutionName", "InstitutionAddress",
    "ReferringPhysicianName", "PerformingPhysicianName",
    "StudyDate", "StudyTime", "AccessionNumber",
    "OtherPatientIDs", "OtherPatientNames",
]

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHEST_XRAY_DIR = os.path.join(DATA_DIR, "chest_xray")
SEGMENTATION_DIR = os.path.join(DATA_DIR, "segmentation")
DICOM_DIR = os.path.join(DATA_DIR, "dicom_samples")
REPORTS_DIR = os.path.join(DATA_DIR, "sample_reports")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
VIZ_DIR = os.path.join(OUTPUT_DIR, "visualizations")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# Model checkpoints
CLASSIFIER_BEST_PATH = os.path.join(MODEL_DIR, "classifier_best.pth")
CLASSIFIER_LAST_PATH = os.path.join(MODEL_DIR, "classifier_last.pth")
SEGMENTER_BEST_PATH = os.path.join(MODEL_DIR, "segmenter_best.pth")
SEGMENTER_LAST_PATH = os.path.join(MODEL_DIR, "segmenter_last.pth")

# ──────────────────────────────────────────────
# Create directories
# ──────────────────────────────────────────────
for _dir in [DATA_DIR, CHEST_XRAY_DIR, SEGMENTATION_DIR, DICOM_DIR,
             REPORTS_DIR, OUTPUT_DIR, MODEL_DIR, VIZ_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)

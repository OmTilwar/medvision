# MedVision: Medical Imaging AI Toolkit

A comprehensive medical imaging toolkit for chest X-ray classification, lung segmentation, DICOM handling, OCR-based de-identification, and model interpretability — built with PyTorch, OpenCV, and FastAPI.

## Key Features

- **Chest X-Ray Classification** — ResNet-18 transfer learning for pneumonia detection (extensible to 14-class ChestX-ray14)
- **Lung Segmentation** — U-Net with Dice+BCE loss for precise lung boundary delineation
- **DICOM Handling** — Load, convert, and visualize clinical DICOM images with proper windowing (lung/bone/brain)
- **PHI De-identification** — HIPAA-compliant removal of Protected Health Information from DICOM files and medical reports
- **OCR Pipeline** — Tesseract-based text extraction from scanned medical documents with entity detection
- **Model Interpretability** — Grad-CAM heatmaps showing disease-specific attention regions
- **REST API** — FastAPI server with 6 endpoints for production-ready model serving
- **Tested** — 45+ unit and integration tests

---

## Architecture

### Classification Model (ResNet-18)

```
Input Image (3 × 224 × 224)
        │
        ▼
┌─────────────────────┐
│  ResNet-18 Backbone  │  ← ImageNet pre-trained (layer1 frozen)
│  Output: 512-dim     │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Classification Head │
│  FC(512→256) → BN    │
│  → ReLU → Dropout    │
│  → FC(256→num_class) │
└─────────────────────┘
        │
        ▼
  Softmax → Disease Probabilities
```

### U-Net Segmentation

```
Input (1 × 256 × 256)         Output (1 × 256 × 256)
        │                              ▲
        ▼                              │
┌──── Encoder ────┐          ┌──── Decoder ────┐
│  64 ──────────────────────────────── 64  │  ← Skip Connection
│  ↓              │          │          ↑  │
│  128 ────────────────────────────── 128  │  ← Skip Connection
│  ↓              │          │          ↑  │
│  256 ────────────────────────────── 256  │  ← Skip Connection
│  ↓              │          │          ↑  │
│  512 ────────────────────────────── 512  │  ← Skip Connection
│  ↓              │          │          ↑  │
│       ┌── Bottleneck: 1024 ──┐       │
└─────────────────┘          └─────────────────┘
```

### OCR + De-identification Pipeline

```
Medical Report Image
        │
        ▼
┌─── Preprocessing ───┐
│  Deskew → Denoise    │
│  → Adaptive Binarize │
└──────────────────────┘
        │
        ▼
┌─── Tesseract OCR ───┐
│  Text Extraction     │
└──────────────────────┘
        │
        ▼
┌─── Entity Detection ───┐
│  Regex NER:              │
│  Names, Dates, MRN,     │
│  Phone, SSN, Email       │
└──────────────────────────┘
        │
        ▼
┌─── De-identification ───┐
│  Replace PHI with        │
│  [REDACTED] placeholders │
└──────────────────────────┘
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/omtilwar/medvision.git
cd medvision
pip install -r requirements.txt
```

### Download Data & Train

```bash
# Set up datasets (downloads or creates synthetic data)
python scripts/download_data.py

# Train chest X-ray classifier (~15 min on GPU)
python -m src.train_classifier

# Train U-Net segmenter (~20 min on GPU)
python -m src.train_segmenter
```

### Run Demo

```bash
python scripts/demo.py    # End-to-end: classify → segment → DICOM → de-identify
```

### Start API Server

```bash
uvicorn api.main:app --reload
```

### Run Tests

```bash
pytest tests/ -v    # 45+ tests
```

---

## API Endpoints

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `POST` | `/classify` | Classify a chest X-ray → disease probabilities |
| `POST` | `/segment` | Segment lung regions → binary mask (PNG) |
| `POST` | `/dicom/convert` | Convert DICOM → PNG + metadata |
| `POST` | `/dicom/anonymize` | De-identify DICOM → anonymized file |
| `POST` | `/deidentify` | OCR + de-identify medical report image |
| `GET` | `/health` | Health check with model status |

### Example API Calls

```bash
# Classify a chest X-ray
curl -X POST -F "file=@xray.png" http://localhost:8000/classify

# Segment lung regions
curl -X POST -F "file=@xray.png" http://localhost:8000/segment -o mask.png

# De-identify a medical report
curl -X POST -F "file=@report.png" http://localhost:8000/deidentify
```

---

## Medical Image Preprocessing

MedVision implements domain-specific preprocessing for clinical imaging:

- **CLAHE** — Contrast Limited Adaptive Histogram Equalization for X-ray enhancement
- **CT Windowing** — Lung, bone, soft tissue, and brain window presets
- **Medical Augmentation** — Anatomically valid transforms (no vertical flip for chest X-rays)
- **Grayscale → 3-channel** — Automatic conversion for pretrained backbone compatibility

---

## DICOM De-identification

Follows **DICOM PS3.15 Annex E** Basic Application Level Confidentiality Profile:

- Removes 25+ PHI tags (PatientName, PatientID, BirthDate, InstitutionName, etc.)
- Generates new UIDs to prevent re-identification via UID matching
- Removes vendor-specific private tags
- Preserves pixel data integrity
- Sets `PatientIdentityRemoved = YES` marker

---

## Project Structure

```
medvision/
├── api/
│   └── main.py                    # FastAPI server (6 endpoints)
├── src/
│   ├── classifier.py              # ResNet-18 chest X-ray classifier
│   ├── segmenter.py               # U-Net lung segmentation
│   ├── dicom_handler.py           # DICOM I/O + anonymization
│   ├── ocr_pipeline.py            # OCR + PHI de-identification
│   ├── interpretability.py        # Grad-CAM for classification
│   ├── preprocessing.py           # CLAHE, windowing, augmentation
│   ├── dataset.py                 # Dataset classes + DataLoaders
│   ├── train_classifier.py        # Classification training loop
│   ├── train_segmenter.py         # Segmentation training loop
│   └── evaluate.py                # AUC-ROC, Dice, confusion matrix
├── scripts/
│   ├── download_data.py           # Dataset setup + synthetic data
│   └── demo.py                    # End-to-end demonstration
├── tests/
│   ├── test_classifier.py         # 10 classifier tests
│   ├── test_segmenter.py          # 13 segmenter + loss tests
│   ├── test_dicom.py              # 8 DICOM tests
│   ├── test_ocr.py                # 14 OCR + de-identification tests
│   └── test_api.py                # 6 API endpoint tests
├── config.py                      # Central configuration
├── requirements.txt               # Dependencies
├── Dockerfile                     # Container deployment
└── README.md
```

---

## Tech Stack

- **PyTorch** — Model architecture and training
- **torchvision** — Pre-trained backbones (ResNet-18)
- **OpenCV** — Image preprocessing (CLAHE, windowing, OCR prep)
- **pydicom** — DICOM file parsing and de-identification
- **pytesseract** — Optical Character Recognition
- **pytorch-grad-cam** — Model interpretability
- **FastAPI** — REST API serving
- **scikit-learn** — Evaluation metrics (AUC-ROC, confusion matrix)
- **seaborn/matplotlib** — Visualization

---

## Clinical Relevance

This toolkit addresses real-world medical imaging challenges:

1. **Disease Detection**: Automated screening can assist radiologists in identifying pneumonia from chest X-rays, reducing diagnostic delays.
2. **Lung Segmentation**: Precise delineation of lung boundaries enables quantitative analysis (lung volume, lesion area measurement).
3. **DICOM Compliance**: Direct handling of clinical DICOM format enables integration with hospital PACS systems.
4. **HIPAA Compliance**: Automated de-identification enables safe sharing of medical data for research while protecting patient privacy.
5. **Explainability**: Grad-CAM visualizations build clinician trust by showing *why* the model made a prediction, not just *what* it predicted.

---

## License

MIT

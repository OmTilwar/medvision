"""
MedVision FastAPI Server
REST API for medical image analysis: classification, segmentation, 
DICOM handling, and de-identification.
"""

import os
import sys
import io
import json
import tempfile
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ──────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────

app = FastAPI(
    title="MedVision API",
    description="Medical Imaging AI Toolkit — Classification, Segmentation, DICOM, & De-identification",
    version="1.0.0",
)

# Lazy-loaded models (loaded on first request)
_classifier = None
_segmenter = None


def get_classifier():
    """Lazy-load the classification model."""
    global _classifier
    if _classifier is None:
        if os.path.exists(config.CLASSIFIER_BEST_PATH):
            try:
                from src.classifier import load_classifier
                _classifier = load_classifier(config.CLASSIFIER_BEST_PATH)
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Classifier model weights not loadable: {str(e)}"
                )
        else:
            raise HTTPException(
                status_code=503,
                detail="Classifier model not found. Train first: python -m src.train_classifier"
            )
    return _classifier


def get_segmenter():
    """Lazy-load the segmentation model."""
    global _segmenter
    if _segmenter is None:
        if os.path.exists(config.SEGMENTER_BEST_PATH):
            try:
                from src.segmenter import load_segmenter
                _segmenter = load_segmenter(config.SEGMENTER_BEST_PATH)
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Segmenter model weights not loadable: {str(e)}"
                )
        else:
            raise HTTPException(
                status_code=503,
                detail="Segmenter model not found. Train first: python -m src.train_segmenter"
            )
    return _segmenter


async def load_image_from_upload(file: UploadFile) -> Image.Image:
    """Read uploaded file into PIL Image."""
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint with model status."""
    import torch
    
    return {
        "status": "healthy",
        "device": str(config.DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "classifier_loaded": _classifier is not None,
        "segmenter_loaded": _segmenter is not None,
        "classifier_model_exists": os.path.exists(config.CLASSIFIER_BEST_PATH),
        "segmenter_model_exists": os.path.exists(config.SEGMENTER_BEST_PATH),
    }


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Classify a chest X-ray image.
    
    Returns disease predictions with confidence scores.
    """
    import torch
    from src.preprocessing import get_classification_test_transforms
    
    model = get_classifier()
    model.eval()
    device = next(model.parameters()).device
    
    # Load and preprocess image
    image = await load_image_from_upload(file)
    image = image.convert("RGB")
    
    transform = get_classification_test_transforms()
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    
    predicted_class = int(probs.argmax())
    predicted_name = config.CLASS_NAMES[predicted_class] if predicted_class < len(config.CLASS_NAMES) else f"Class {predicted_class}"
    
    return {
        "prediction": predicted_name,
        "confidence": float(probs[predicted_class]),
        "probabilities": {
            name: float(probs[i])
            for i, name in enumerate(config.CLASS_NAMES)
            if i < len(probs)
        },
        "model": config.CLASSIFIER_BACKBONE,
    }


@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    """
    Segment lung regions in a chest X-ray.
    
    Returns the segmentation mask as a PNG image.
    """
    import torch
    from src.preprocessing import get_segmentation_test_transforms
    
    model = get_segmenter()
    model.eval()
    device = next(model.parameters()).device
    
    # Load and preprocess image
    image = await load_image_from_upload(file)
    
    img_transform, _ = get_segmentation_test_transforms()
    input_tensor = img_transform(image).unsqueeze(0).to(device)
    
    # Handle channel mismatch (model expects 1ch, transforms give 3ch)
    if input_tensor.shape[1] == 3 and config.UNET_IN_CHANNELS == 1:
        # Convert to grayscale by averaging channels
        input_tensor = input_tensor.mean(dim=1, keepdim=True)
    
    # Predict
    with torch.no_grad():
        logits = model(input_tensor)
        mask = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    # Threshold
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Convert to PNG
    mask_image = Image.fromarray(binary_mask)
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="image/png",
        headers={"X-Lung-Coverage": f"{(mask > 0.5).mean():.2%}"}
    )


@app.post("/dicom/convert")
async def convert_dicom(file: UploadFile = File(...)):
    """
    Convert a DICOM file to PNG with metadata extraction.
    
    Returns metadata JSON. The converted image can be retrieved separately.
    """
    from src.dicom_handler import load_dicom, extract_metadata
    
    # Save uploaded DICOM to temp file
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    
    try:
        pixel_array, metadata = load_dicom(tmp_path)
        
        # Normalize for display
        pmin, pmax = pixel_array.min(), pixel_array.max()
        if pmax > pmin:
            display_image = ((pixel_array - pmin) / (pmax - pmin) * 255).astype(np.uint8)
        else:
            display_image = np.zeros_like(pixel_array, dtype=np.uint8)
        
        return {
            "metadata": metadata,
            "image_shape": list(pixel_array.shape),
            "pixel_range": {"min": float(pmin), "max": float(pmax)},
            "filename": file.filename,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid DICOM file: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/dicom/anonymize")
async def anonymize_dicom_endpoint(file: UploadFile = File(...)):
    """
    Anonymize a DICOM file by removing Protected Health Information.
    
    Returns the anonymized DICOM file.
    """
    from src.dicom_handler import anonymize_dicom
    
    # Save uploaded DICOM
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    
    try:
        # Anonymize
        anon_path = tmp_path.replace(".dcm", "_anon.dcm")
        removed_phi = anonymize_dicom(tmp_path, output_path=anon_path)
        
        # Return anonymized file
        with open(anon_path, "rb") as f:
            anon_contents = f.read()
        
        return StreamingResponse(
            io.BytesIO(anon_contents),
            media_type="application/dicom",
            headers={
                "Content-Disposition": f"attachment; filename=anonymized_{file.filename}",
                "X-PHI-Tags-Removed": str(len(removed_phi)),
            }
        )
    finally:
        # Cleanup
        for path in [tmp_path, anon_path]:
            if os.path.exists(path):
                os.unlink(path)


@app.post("/deidentify")
async def deidentify_report(file: UploadFile = File(...)):
    """
    De-identify a medical report image.
    
    Performs OCR, detects PHI entities, and returns de-identified text.
    """
    from src.ocr_pipeline import extract_text, deidentify_text
    
    image = await load_image_from_upload(file)
    image_np = np.array(image)
    
    try:
        # OCR
        text = extract_text(image_np, preprocess=True)
        
        # De-identify
        result = deidentify_text(text)
        
        return {
            "original_text": result.original_text,
            "deidentified_text": result.deidentified_text,
            "entities_found": [
                {
                    "type": e.entity_type,
                    "value": e.value,
                    "start": e.start,
                    "end": e.end,
                }
                for e in result.entities_found
            ],
            "entity_count": result.entity_count,
        }
    except (RuntimeError, FileNotFoundError, OSError) as e:
        raise HTTPException(status_code=503, detail=f"OCR dependency error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

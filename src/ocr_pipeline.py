"""
MedVision OCR & De-identification Pipeline
Extracts text from medical report images and removes Protected Health Information (PHI).
Designed for clinical document processing workflows.

Uses Tesseract OCR with medical-specific preprocessing and regex-based NER
for entity detection and de-identification.
"""

import os
import re
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# Conditional import — Tesseract may not be installed
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("[WARN] pytesseract not installed. OCR features disabled. Install: pip install pytesseract")


# ──────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────

@dataclass
class DetectedEntity:
    """A detected PHI entity in medical text."""
    entity_type: str       # e.g., "PATIENT_NAME", "DATE", "MRN", "PHONE"
    value: str             # The matched text
    start: int             # Start position in text
    end: int               # End position in text
    confidence: float = 1.0  # Detection confidence (1.0 for regex, <1.0 for ML)


@dataclass
class DeidentificationResult:
    """Result of de-identification process."""
    original_text: str
    deidentified_text: str
    entities_found: List[DetectedEntity] = field(default_factory=list)
    entity_count: Dict[str, int] = field(default_factory=dict)


# ──────────────────────────────────────────────
# OCR Preprocessing
# ──────────────────────────────────────────────

def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocess medical document image for optimal OCR accuracy.
    
    Pipeline:
        1. Convert to grayscale
        2. Resize if too small (minimum 300 DPI equivalent)
        3. Deskew (straighten rotated documents)
        4. Denoise (reduce scanning artifacts)
        5. Adaptive binarization (handle uneven illumination)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Resize if too small (Tesseract works best with ~300 DPI)
    h, w = gray.shape
    if max(h, w) < 1000:
        scale = 1000 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Adaptive thresholding (handles uneven illumination in scanned documents)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    return binary


def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Deskew (straighten) a rotated document image.
    Uses Hough line detection to find dominant text angle.
    """
    # Edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    
    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return image
    
    # Calculate median angle
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 45:  # Only consider near-horizontal lines
            angles.append(angle)
    
    if not angles:
        return image
    
    median_angle = np.median(angles)
    
    # Rotate to correct
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


# ──────────────────────────────────────────────
# OCR Text Extraction
# ──────────────────────────────────────────────

def extract_text(
    image_input,
    preprocess: bool = True,
    lang: str = "eng",
    config_str: str = "--oem 3 --psm 6",
) -> str:
    """
    Extract text from a medical document image using Tesseract OCR.
    
    Args:
        image_input: Image path, PIL Image, or numpy array
        preprocess: Whether to apply medical document preprocessing
        lang: Tesseract language code
        config_str: Tesseract configuration
            --oem 3: LSTM neural net engine
            --psm 6: Assume uniform block of text
            
    Returns:
        Extracted text string
    """
    if not TESSERACT_AVAILABLE:
        raise RuntimeError(
            "pytesseract is required. Install: pip install pytesseract\n"
            "Also install Tesseract OCR: https://github.com/tesseract-ocr/tesseract"
        )
    
    # Load image
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_input}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        image = np.array(image_input)
    elif isinstance(image_input, np.ndarray):
        image = image_input.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")
    
    # Preprocess for better OCR accuracy
    if preprocess:
        image = preprocess_for_ocr(image)
    
    # Run Tesseract
    text = pytesseract.image_to_string(image, lang=lang, config=config_str)
    
    return text.strip()


def extract_text_with_boxes(
    image_input,
    preprocess: bool = True,
) -> List[Dict[str, Any]]:
    """
    Extract text with bounding box coordinates.
    Useful for redacting PHI directly on the image.
    
    Returns:
        List of dicts with 'text', 'x', 'y', 'w', 'h', 'conf' keys
    """
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("pytesseract is required for OCR.")
    
    # Load image
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        image = np.array(image_input)
    else:
        image = image_input.copy()
    
    if preprocess:
        image = preprocess_for_ocr(image)
    
    # Get word-level data with positions
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    results = []
    for i in range(len(data['text'])):
        if data['text'][i].strip():
            results.append({
                'text': data['text'][i],
                'x': data['left'][i],
                'y': data['top'][i],
                'w': data['width'][i],
                'h': data['height'][i],
                'conf': int(data['conf'][i]),
            })
    
    return results


# ──────────────────────────────────────────────
# PHI Entity Detection (Regex-based NER)
# ──────────────────────────────────────────────

# Compiled regex patterns for common PHI entities
PHI_PATTERNS = {
    "DATE": [
        # MM/DD/YYYY, MM-DD-YYYY
        re.compile(r'\b(0?[1-9]|1[0-2])[/\-](0?[1-9]|[12]\d|3[01])[/\-](\d{4})\b'),
        # DD-Mon-YYYY (e.g., 15-Mar-2024)
        re.compile(r'\b(\d{1,2})[/\-](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[/\-](\d{4})\b', re.I),
        # YYYY-MM-DD (ISO)
        re.compile(r'\b(\d{4})[/\-](0?[1-9]|1[0-2])[/\-](0?[1-9]|[12]\d|3[01])\b'),
        # Month DD, YYYY
        re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', re.I),
    ],
    "MRN": [
        # Medical Record Numbers — common patterns
        re.compile(r'\bMRN[:\s#]*(\d{4,12})\b', re.I),
        re.compile(r'\bMedical Record[:\s#]*(\d{4,12})\b', re.I),
        re.compile(r'\bPatient ID[:\s#]*([A-Z0-9]{4,12})\b', re.I),
    ],
    "PHONE": [
        # US phone numbers
        re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
        # International format
        re.compile(r'\b\+\d{1,3}[-.\s]?\d{4,14}\b'),
    ],
    "SSN": [
        # Social Security Numbers
        re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'),
    ],
    "EMAIL": [
        re.compile(r'\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b'),
    ],
    "AGE": [
        # Age patterns (e.g., "45 years old", "45yo", "age: 45")
        re.compile(r'\b(\d{1,3})\s*(?:years?\s*old|y/?o|yo)\b', re.I),
        re.compile(r'\bage[:\s]*(\d{1,3})\b', re.I),
    ],
    "DOCTOR_NAME": [
        # Dr. Firstname Lastname
        re.compile(r'\bDr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b'),
        # Physician: Name
        re.compile(r'\b(?:Physician|Doctor|Attending|Referring)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b'),
    ],
    "PATIENT_NAME": [
        # Patient: Firstname Lastname
        re.compile(r'\bPatient(?:\s+Name)?[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b'),
        # Name: Firstname Lastname
        re.compile(r'\bName[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b'),
    ],
}

# Replacement placeholders for each entity type
REPLACEMENT_MAP = {
    "DATE": "[DATE_REDACTED]",
    "MRN": "[MRN_REDACTED]",
    "PHONE": "[PHONE_REDACTED]",
    "SSN": "[SSN_REDACTED]",
    "EMAIL": "[EMAIL_REDACTED]",
    "AGE": "[AGE_REDACTED]",
    "DOCTOR_NAME": "[DOCTOR_REDACTED]",
    "PATIENT_NAME": "[PATIENT_REDACTED]",
}


def detect_entities(text: str) -> List[DetectedEntity]:
    """
    Detect PHI entities in medical text using regex pattern matching.
    
    Detects:
        - Patient names, doctor names
        - Dates (multiple formats)
        - Medical Record Numbers (MRN)
        - Phone numbers, SSNs, email addresses
        - Age references
    
    Args:
        text: Input text from OCR or direct input
        
    Returns:
        List of DetectedEntity objects with positions and types
    """
    entities = []
    
    for entity_type, patterns in PHI_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                entities.append(DetectedEntity(
                    entity_type=entity_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0,
                ))
    
    # Sort by position (start index)
    entities.sort(key=lambda e: e.start)
    
    # Remove overlapping entities (keep longer match)
    filtered = []
    for entity in entities:
        if filtered and entity.start < filtered[-1].end:
            # Overlapping — keep the longer one
            if entity.end - entity.start > filtered[-1].end - filtered[-1].start:
                filtered[-1] = entity
        else:
            filtered.append(entity)
    
    return filtered


def deidentify_text(text: str) -> DeidentificationResult:
    """
    De-identify medical text by replacing PHI entities with placeholders.
    
    Args:
        text: Input medical text
        
    Returns:
        DeidentificationResult with original text, de-identified text, and entity list
    """
    entities = detect_entities(text)
    
    # Replace entities in reverse order to preserve positions
    deidentified = text
    for entity in reversed(entities):
        replacement = REPLACEMENT_MAP.get(entity.entity_type, "[REDACTED]")
        deidentified = (
            deidentified[:entity.start] + replacement + deidentified[entity.end:]
        )
    
    # Count entities by type
    entity_count = {}
    for entity in entities:
        entity_count[entity.entity_type] = entity_count.get(entity.entity_type, 0) + 1
    
    return DeidentificationResult(
        original_text=text,
        deidentified_text=deidentified,
        entities_found=entities,
        entity_count=entity_count,
    )


def deidentify_image(
    image_input,
    output_path: Optional[str] = None,
) -> Tuple[np.ndarray, DeidentificationResult]:
    """
    De-identify a medical report image by:
    1. Running OCR to extract text
    2. Detecting PHI entities
    3. Drawing black rectangles over PHI regions in the image
    
    Args:
        image_input: Image path, PIL Image, or numpy array
        output_path: Where to save the redacted image
        
    Returns:
        (redacted_image, deidentification_result)
    """
    # Load image
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        image = np.array(image_input)
    else:
        image = image_input.copy()
    
    # Get text with bounding boxes
    word_data = extract_text_with_boxes(image_input, preprocess=False)
    
    # Get full text and detect entities
    full_text = extract_text(image_input)
    result = deidentify_text(full_text)
    
    # Build word position map
    redacted_image = image.copy()
    
    # For each entity, find matching words and redact
    for entity in result.entities_found:
        entity_words = entity.value.lower().split()
        
        for word_info in word_data:
            if word_info['text'].lower() in entity_words:
                # Draw black rectangle over the word
                x, y, w, h = word_info['x'], word_info['y'], word_info['w'], word_info['h']
                padding = 3
                cv2.rectangle(
                    redacted_image,
                    (x - padding, y - padding),
                    (x + w + padding, y + h + padding),
                    (0, 0, 0),  # Black
                    -1,  # Filled
                )
    
    if output_path:
        Image.fromarray(redacted_image).save(output_path)
        print(f"Redacted image saved: {output_path}")
    
    return redacted_image, result


def process_report(
    image_input,
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full medical report processing pipeline:
    1. OCR text extraction
    2. PHI entity detection
    3. Text de-identification
    4. Image redaction
    
    Args:
        image_input: Path to medical report image
        save_dir: Directory to save outputs
        
    Returns:
        Processing results dict
    """
    save_dir = save_dir or config.OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("  Medical Report Processing Pipeline")
    print("=" * 60)
    
    # Step 1: OCR
    print("\n[1/4] Extracting text via OCR...")
    raw_text = extract_text(image_input)
    print(f"  Extracted {len(raw_text)} characters")
    
    # Step 2: Entity detection
    print("\n[2/4] Detecting PHI entities...")
    entities = detect_entities(raw_text)
    print(f"  Found {len(entities)} entities:")
    for entity in entities:
        print(f"    - {entity.entity_type}: '{entity.value}'")
    
    # Step 3: Text de-identification
    print("\n[3/4] De-identifying text...")
    deident_result = deidentify_text(raw_text)
    
    # Step 4: Image redaction
    print("\n[4/4] Redacting image...")
    redacted_path = os.path.join(save_dir, "redacted_report.png")
    redacted_image, _ = deidentify_image(image_input, output_path=redacted_path)
    
    # Save text outputs
    original_text_path = os.path.join(save_dir, "original_text.txt")
    deident_text_path = os.path.join(save_dir, "deidentified_text.txt")
    
    with open(original_text_path, 'w') as f:
        f.write(raw_text)
    with open(deident_text_path, 'w') as f:
        f.write(deident_result.deidentified_text)
    
    print(f"\n{'='*60}")
    print(f"  Processing complete!")
    print(f"  Entities found: {deident_result.entity_count}")
    print(f"  Redacted image: {redacted_path}")
    print(f"{'='*60}")
    
    return {
        "original_text": raw_text,
        "deidentified_text": deident_result.deidentified_text,
        "entities": [
            {"type": e.entity_type, "value": e.value, "start": e.start, "end": e.end}
            for e in entities
        ],
        "entity_count": deident_result.entity_count,
        "redacted_image_path": redacted_path,
    }

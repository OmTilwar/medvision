"""
MedVision End-to-End Demo
Demonstrates all capabilities: classification, segmentation, DICOM handling, 
OCR/de-identification, and Grad-CAM interpretability.
"""

import os
import sys
import json

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def demo_classification():
    """Demo: Chest X-ray classification with Grad-CAM."""
    print("\n" + "=" * 60)
    print("  Demo 1: Chest X-Ray Classification + Grad-CAM")
    print("=" * 60)
    
    from src.classifier import build_classifier
    from src.preprocessing import get_classification_test_transforms
    
    # Build model (untrained — just shows the pipeline)
    model = build_classifier()
    model.eval()
    device = next(model.parameters()).device
    
    # Find a test image
    test_dir = os.path.join(config.CHEST_XRAY_DIR, "test")
    test_image = None
    
    for class_dir in os.listdir(test_dir) if os.path.exists(test_dir) else []:
        class_path = os.path.join(test_dir, class_dir)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                test_image = os.path.join(class_path, images[0])
                break
    
    if test_image is None:
        print("  No test images found. Run download_data.py first.")
        return
    
    print(f"  Test image: {test_image}")
    
    # Classify
    transform = get_classification_test_transforms()
    image = Image.open(test_image).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    
    predicted = probs.argmax()
    print(f"  Prediction: {config.CLASS_NAMES[predicted]} ({probs[predicted]:.1%})")
    
    # Grad-CAM
    try:
        from src.interpretability import visualize_prediction
        save_path = os.path.join(config.VIZ_DIR, "demo_classification.png")
        visualize_prediction(model, test_image, save_path=save_path)
        print(f"  Grad-CAM visualization saved: {save_path}")
    except Exception as e:
        print(f"  [WARN] Grad-CAM skipped: {e}")


def demo_segmentation():
    """Demo: Lung segmentation with U-Net."""
    print("\n" + "=" * 60)
    print("  Demo 2: Lung Segmentation (U-Net)")
    print("=" * 60)
    
    from src.segmenter import build_segmenter
    
    # Build model
    model = build_segmenter()
    model.eval()
    device = next(model.parameters()).device
    
    # Create a test input
    test_input = torch.randn(1, config.UNET_IN_CHANNELS, config.SEG_IMAGE_SIZE, config.SEG_IMAGE_SIZE).to(device)
    
    with torch.no_grad():
        output = model(test_input)
        mask = torch.sigmoid(output)
    
    print(f"  Input shape:  {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Mask range:   [{mask.min():.4f}, {mask.max():.4f}]")
    print(f"  Lung coverage: {(mask > 0.5).float().mean():.1%}")
    
    # Save sample mask
    mask_np = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_np)
    save_path = os.path.join(config.VIZ_DIR, "demo_segmentation_mask.png")
    mask_image.save(save_path)
    print(f"  Sample mask saved: {save_path}")


def demo_dicom():
    """Demo: DICOM handling and de-identification."""
    print("\n" + "=" * 60)
    print("  Demo 3: DICOM Handling & De-identification")
    print("=" * 60)
    
    from src.dicom_handler import PYDICOM_AVAILABLE
    
    if not PYDICOM_AVAILABLE:
        print("  [SKIP] pydicom not installed")
        return
    
    from src.dicom_handler import create_sample_dicom, load_dicom, extract_metadata, anonymize_dicom
    
    # Create sample DICOM
    sample_path = os.path.join(config.DICOM_DIR, "demo_sample.dcm")
    create_sample_dicom(sample_path, patient_name="Demo^Patient")
    
    # Load and display metadata
    pixel_array, metadata = load_dicom(sample_path)
    print(f"\n  DICOM Metadata:")
    for key, value in metadata.items():
        if value != "N/A":
            print(f"    {key}: {value}")
    
    print(f"\n  Pixel array shape: {pixel_array.shape}")
    print(f"  Pixel range: [{pixel_array.min():.0f}, {pixel_array.max():.0f}]")
    
    # Anonymize
    anon_path = os.path.join(config.DICOM_DIR, "demo_sample_anon.dcm")
    removed_phi = anonymize_dicom(sample_path, output_path=anon_path)
    
    print(f"\n  De-identification results:")
    for tag, original_value in removed_phi.items():
        print(f"    {tag}: '{original_value}' → [REMOVED]")
    
    # Verify anonymization
    _, anon_metadata = load_dicom(anon_path)
    print(f"\n  Anonymized patient name: {anon_metadata['patient_name']}")
    print(f"  Anonymized patient ID: {anon_metadata['patient_id']}")


def demo_ocr():
    """Demo: OCR and de-identification of medical reports."""
    print("\n" + "=" * 60)
    print("  Demo 4: OCR & PHI De-identification")
    print("=" * 60)
    
    from src.ocr_pipeline import detect_entities, deidentify_text, TESSERACT_AVAILABLE
    
    # Demo with text input (works without Tesseract)
    sample_text = """
    RADIOLOGY REPORT
    
    Patient Name: John Smith
    Patient ID: MRN123456
    Date of Birth: 03/15/1985
    Date of Exam: 01/15/2024
    Referring Physician: Dr. Sarah Johnson
    Phone: (555) 123-4567
    Email: john.smith@email.com
    
    EXAMINATION: Chest PA and Lateral
    
    CLINICAL INDICATION:
    45 years old male with persistent cough.
    
    FINDINGS:
    The lungs are clear bilaterally. No focal consolidation,
    pleural effusion, or pneumothorax.
    
    IMPRESSION: Normal chest radiograph.
    
    Radiologist: Dr. Michael Brown
    Institution: Sample Medical Center
    """
    
    # Detect PHI entities
    entities = detect_entities(sample_text)
    print(f"\n  PHI Entities Detected ({len(entities)}):")
    for entity in entities:
        print(f"    [{entity.entity_type}] '{entity.value}'")
    
    # De-identify
    result = deidentify_text(sample_text)
    print(f"\n  De-identified text (first 300 chars):")
    print(f"  {result.deidentified_text[:300]}...")
    
    print(f"\n  Entity counts: {result.entity_count}")
    
    if TESSERACT_AVAILABLE:
        # Demo with image OCR
        report_path = os.path.join(config.REPORTS_DIR, "sample_report.png")
        if os.path.exists(report_path):
            from src.ocr_pipeline import extract_text
            ocr_text = extract_text(report_path)
            print(f"\n  OCR extracted {len(ocr_text)} characters from sample report image")
    else:
        print("\n  [NOTE] Install Tesseract for image-based OCR demo")


def main():
    """Run all demos."""
    print("=" * 60)
    print("  MedVision — End-to-End Demo")
    print("=" * 60)
    print(f"  Device: {config.DEVICE}")
    
    demo_classification()
    demo_segmentation()
    demo_dicom()
    demo_ocr()
    
    print("\n" + "=" * 60)
    print("  All demos completed!")
    print("=" * 60)
    print(f"  Visualizations saved to: {config.VIZ_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

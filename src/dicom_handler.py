"""
MedVision DICOM Handler
DICOM file parsing, visualization, conversion, and de-identification.
Handles clinical image formats as specified in medical imaging workflows.

Uses pydicom for DICOM parsing — the standard library for medical imaging I/O.
"""

import os
import sys
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# Conditional import — pydicom may not be installed in all environments
try:
    import pydicom
    from pydicom.dataset import Dataset as DicomDataset
    from pydicom.uid import generate_uid
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("[WARN] pydicom not installed. DICOM features disabled. Install: pip install pydicom")


# ──────────────────────────────────────────────
# DICOM Loading & Conversion
# ──────────────────────────────────────────────

def load_dicom(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a DICOM file and return pixel array + metadata.
    
    Handles:
        - Rescale Slope/Intercept (converts stored values → Hounsfield Units)
        - Photometric Interpretation (MONOCHROME1 vs MONOCHROME2)
        - Window Center/Width (applies clinical windowing if present)
    
    Args:
        filepath: Path to .dcm file
        
    Returns:
        (pixel_array, metadata_dict)
    """
    if not PYDICOM_AVAILABLE:
        raise RuntimeError("pydicom is required for DICOM operations. Install: pip install pydicom")
    
    ds = pydicom.dcmread(filepath)
    
    # Extract pixel data
    pixel_array = ds.pixel_array.astype(np.float64)
    
    # Apply Rescale Slope/Intercept (raw stored values → physical values like HU)
    slope = getattr(ds, 'RescaleSlope', 1.0)
    intercept = getattr(ds, 'RescaleIntercept', 0.0)
    pixel_array = pixel_array * float(slope) + float(intercept)
    
    # Handle Photometric Interpretation
    # MONOCHROME1: white = 0 (inverted), MONOCHROME2: white = max (standard)
    photometric = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
    if photometric == 'MONOCHROME1':
        pixel_array = pixel_array.max() - pixel_array
    
    # Extract key metadata
    metadata = extract_metadata(ds)
    
    return pixel_array, metadata


def extract_metadata(ds_or_path) -> Dict[str, Any]:
    """
    Extract clinically relevant metadata from a DICOM dataset.
    
    Returns structured dict with patient, study, and imaging info.
    """
    if not PYDICOM_AVAILABLE:
        raise RuntimeError("pydicom is required. Install: pip install pydicom")
    
    if isinstance(ds_or_path, str):
        ds = pydicom.dcmread(ds_or_path, stop_before_pixels=True)
    else:
        ds = ds_or_path
    
    def safe_get(attr: str, default: str = "N/A") -> str:
        val = getattr(ds, attr, default)
        return str(val) if val else default
    
    return {
        # Patient info
        "patient_name": safe_get("PatientName"),
        "patient_id": safe_get("PatientID"),
        "patient_birth_date": safe_get("PatientBirthDate"),
        "patient_sex": safe_get("PatientSex"),
        "patient_age": safe_get("PatientAge"),
        
        # Study info
        "study_date": safe_get("StudyDate"),
        "study_time": safe_get("StudyTime"),
        "study_description": safe_get("StudyDescription"),
        "accession_number": safe_get("AccessionNumber"),
        
        # Series info
        "modality": safe_get("Modality"),
        "series_description": safe_get("SeriesDescription"),
        "body_part": safe_get("BodyPartExamined"),
        
        # Image info
        "rows": safe_get("Rows"),
        "columns": safe_get("Columns"),
        "bits_stored": safe_get("BitsStored"),
        "photometric_interpretation": safe_get("PhotometricInterpretation"),
        "pixel_spacing": safe_get("PixelSpacing"),
        
        # Institution
        "institution_name": safe_get("InstitutionName"),
        "referring_physician": safe_get("ReferringPhysicianName"),
    }


def dicom_to_png(
    filepath: str,
    output_path: Optional[str] = None,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
) -> np.ndarray:
    """
    Convert a DICOM file to a displayable PNG image.
    
    Applies windowing (if specified or found in DICOM headers) and 
    normalizes to 8-bit for display.
    
    Args:
        filepath: Path to .dcm file
        output_path: If provided, save PNG to this path
        window_center: Override window center (HU)
        window_width: Override window width (HU)
        
    Returns:
        8-bit numpy array (H, W)
    """
    pixel_array, metadata = load_dicom(filepath)
    
    # Try to get windowing from DICOM headers if not specified
    if window_center is None or window_width is None:
        ds = pydicom.dcmread(filepath, stop_before_pixels=True)
        wc = getattr(ds, 'WindowCenter', None)
        ww = getattr(ds, 'WindowWidth', None)
        
        if wc is not None and ww is not None:
            # Handle multi-valued windows (take first)
            window_center = float(wc[0]) if hasattr(wc, '__iter__') else float(wc)
            window_width = float(ww[0]) if hasattr(ww, '__iter__') else float(ww)
    
    if window_center is not None and window_width is not None:
        # Apply clinical windowing
        from src.preprocessing import CTWindowing
        windower = CTWindowing(int(window_center), int(window_width))
        image_8bit = windower(pixel_array)
    else:
        # Simple min-max normalization
        pmin, pmax = pixel_array.min(), pixel_array.max()
        if pmax > pmin:
            image_8bit = ((pixel_array - pmin) / (pmax - pmin) * 255).astype(np.uint8)
        else:
            image_8bit = np.zeros_like(pixel_array, dtype=np.uint8)
    
    if output_path:
        Image.fromarray(image_8bit).save(output_path)
        print(f"Saved PNG: {output_path}")
    
    return image_8bit


# ──────────────────────────────────────────────
# DICOM De-identification (PHI Removal)
# ──────────────────────────────────────────────

# DICOM tags containing Protected Health Information (PHI)
# Based on DICOM PS3.15 Annex E — Basic Application Level Confidentiality Profile
PHI_TAGS_TO_REMOVE = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "ReferringPhysicianAddress",
    "ReferringPhysicianTelephoneNumbers",
    "PerformingPhysicianName",
    "NameOfPhysiciansReadingStudy",
    "OperatorsName",
    "OtherPatientIDs",
    "OtherPatientNames",
    "MedicalRecordLocator",
    "EthnicGroup",
    "Occupation",
    "AdditionalPatientHistory",
    "PatientComments",
    "StudyDate",
    "StudyTime",
    "AccessionNumber",
    "StudyID",
    "RequestingPhysician",
]

# Replacement values for de-identified tags
DEIDENTIFIED_REPLACEMENTS = {
    "PatientName": "ANONYMOUS",
    "PatientID": "000000",
    "PatientBirthDate": "19000101",
    "InstitutionName": "DEIDENTIFIED",
    "ReferringPhysicianName": "DEIDENTIFIED",
    "AccessionNumber": "000000",
    "StudyDate": "19000101",
    "StudyTime": "000000",
}


def anonymize_dicom(
    filepath: str,
    output_path: Optional[str] = None,
    remove_private_tags: bool = True,
) -> Dict[str, str]:
    """
    De-identify a DICOM file by removing/replacing PHI tags.
    
    Follows DICOM PS3.15 Annex E Basic Application Level Confidentiality Profile.
    This is critical for HIPAA compliance in medical imaging pipelines.
    
    Args:
        filepath: Path to input .dcm file
        output_path: Path to save anonymized .dcm file (default: adds '_anon' suffix)
        remove_private_tags: If True, remove all private/vendor-specific tags
        
    Returns:
        Dict mapping tag names to their original values (for audit logging)
    """
    if not PYDICOM_AVAILABLE:
        raise RuntimeError("pydicom is required. Install: pip install pydicom")
    
    ds = pydicom.dcmread(filepath)
    
    # Track removed PHI for audit
    removed_phi = {}
    
    # Remove/replace PHI tags
    for tag_name in PHI_TAGS_TO_REMOVE:
        if hasattr(ds, tag_name):
            original_value = str(getattr(ds, tag_name))
            removed_phi[tag_name] = original_value
            
            if tag_name in DEIDENTIFIED_REPLACEMENTS:
                setattr(ds, tag_name, DEIDENTIFIED_REPLACEMENTS[tag_name])
            else:
                delattr(ds, tag_name)
    
    # Remove private tags (vendor-specific, may contain PHI)
    if remove_private_tags:
        ds.remove_private_tags()
    
    # Generate new UIDs to prevent re-identification via UID matching
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    
    # Add de-identification marker
    ds.PatientIdentityRemoved = "YES"
    
    # Save anonymized file
    if output_path is None:
        stem, ext = os.path.splitext(filepath)
        output_path = f"{stem}_anon{ext}"
    
    ds.save_as(output_path)
    
    print(f"De-identified DICOM saved: {output_path}")
    print(f"  PHI tags processed: {len(removed_phi)}")
    
    return removed_phi


def batch_process_dicoms(
    input_dir: str,
    output_dir: Optional[str] = None,
    convert_to_png: bool = True,
    anonymize: bool = True,
) -> Dict[str, Any]:
    """
    Batch process a directory of DICOM files.
    
    Performs:
        1. DICOM → PNG conversion (if convert_to_png=True)
        2. DICOM anonymization (if anonymize=True)
        3. Metadata extraction
    
    Args:
        input_dir: Directory containing .dcm files
        output_dir: Output directory (default: input_dir/processed/)
        convert_to_png: Convert DICOMs to PNG images
        anonymize: De-identify DICOMs
    
    Returns:
        Processing summary dict
    """
    output_dir = output_dir or os.path.join(input_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    dcm_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith('.dcm')
    ]
    
    results = {
        "total_files": len(dcm_files),
        "converted": 0,
        "anonymized": 0,
        "errors": [],
        "metadata": [],
    }
    
    for dcm_file in dcm_files:
        filepath = os.path.join(input_dir, dcm_file)
        stem = os.path.splitext(dcm_file)[0]
        
        try:
            # Extract metadata
            metadata = extract_metadata(filepath)
            results["metadata"].append(metadata)
            
            # Convert to PNG
            if convert_to_png:
                png_path = os.path.join(output_dir, f"{stem}.png")
                dicom_to_png(filepath, output_path=png_path)
                results["converted"] += 1
            
            # Anonymize
            if anonymize:
                anon_path = os.path.join(output_dir, f"{stem}_anon.dcm")
                anonymize_dicom(filepath, output_path=anon_path)
                results["anonymized"] += 1
            
        except Exception as e:
            results["errors"].append({"file": dcm_file, "error": str(e)})
            print(f"[ERROR] Failed to process {dcm_file}: {e}")
    
    print(f"\nBatch processing complete:")
    print(f"  Total:      {results['total_files']}")
    print(f"  Converted:  {results['converted']}")
    print(f"  Anonymized: {results['anonymized']}")
    print(f"  Errors:     {len(results['errors'])}")
    
    return results


def create_sample_dicom(
    output_path: str,
    image_size: Tuple[int, int] = (256, 256),
    patient_name: str = "Test^Patient",
    modality: str = "CR",
) -> str:
    """
    Create a sample DICOM file for testing purposes.
    Generates a synthetic chest X-ray-like image with proper DICOM metadata.
    
    Returns:
        Path to created DICOM file
    """
    if not PYDICOM_AVAILABLE:
        raise RuntimeError("pydicom is required. Install: pip install pydicom")
    
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian
    import datetime
    
    # Create file meta
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"  # CR Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    
    # Create dataset
    ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    
    # Patient info (for de-identification testing)
    ds.PatientName = patient_name
    ds.PatientID = "MRN123456"
    ds.PatientBirthDate = "19850315"
    ds.PatientSex = "M"
    ds.PatientAge = "041Y"
    
    # Study info
    ds.StudyDate = datetime.datetime.now().strftime("%Y%m%d")
    ds.StudyTime = datetime.datetime.now().strftime("%H%M%S")
    ds.StudyDescription = "CHEST PA AND LATERAL"
    ds.AccessionNumber = "ACC987654"
    ds.InstitutionName = "Sample Medical Center"
    ds.ReferringPhysicianName = "Dr^Smith^John"
    
    # Series/image info
    ds.Modality = modality
    ds.BodyPartExamined = "CHEST"
    ds.Rows = image_size[0]
    ds.Columns = image_size[1]
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    ds.SOPInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    
    # Generate synthetic X-ray-like pixel data
    np.random.seed(42)
    # Create gradient background simulating chest X-ray
    y_coords = np.linspace(0, 1, image_size[0])
    x_coords = np.linspace(0, 1, image_size[1])
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    # Simulate lung fields (darker elliptical regions)
    lung_left = np.exp(-((xx - 0.35)**2 / 0.02 + (yy - 0.45)**2 / 0.06))
    lung_right = np.exp(-((xx - 0.65)**2 / 0.02 + (yy - 0.45)**2 / 0.06))
    
    # Background + lungs
    pixel_data = (1.0 - 0.6 * lung_left - 0.6 * lung_right) * 2000 + 500
    pixel_data += np.random.normal(0, 50, image_size)
    pixel_data = np.clip(pixel_data, 0, 4095).astype(np.uint16)
    
    ds.PixelData = pixel_data.tobytes()
    
    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ds.save_as(output_path)
    
    print(f"Created sample DICOM: {output_path}")
    return output_path

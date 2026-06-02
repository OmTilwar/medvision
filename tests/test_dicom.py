"""
Tests for MedVision DICOM Handler
Tests DICOM creation, loading, metadata extraction, and de-identification.
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# Skip all tests if pydicom is not available
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

from src.dicom_handler import PYDICOM_AVAILABLE as MODULE_AVAILABLE


@pytest.mark.skipif(not PYDICOM_AVAILABLE, reason="pydicom not installed")
class TestDICOMHandler:
    """Test suite for DICOM handling."""
    
    @pytest.fixture
    def sample_dicom_path(self, tmp_path):
        """Create a sample DICOM file for testing."""
        from src.dicom_handler import create_sample_dicom
        
        output_path = str(tmp_path / "test_sample.dcm")
        create_sample_dicom(
            output_path=output_path,
            patient_name="Test^Patient",
            modality="CR",
        )
        return output_path
    
    def test_create_sample_dicom(self, tmp_path):
        """Sample DICOM creation should produce a valid file."""
        from src.dicom_handler import create_sample_dicom
        
        output_path = str(tmp_path / "test.dcm")
        result = create_sample_dicom(output_path)
        
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0
    
    def test_load_dicom(self, sample_dicom_path):
        """Loading a DICOM should return pixel array and metadata."""
        from src.dicom_handler import load_dicom
        
        pixel_array, metadata = load_dicom(sample_dicom_path)
        
        assert isinstance(pixel_array, np.ndarray)
        assert pixel_array.shape == (256, 256)
        assert isinstance(metadata, dict)
        assert "patient_name" in metadata
    
    def test_extract_metadata(self, sample_dicom_path):
        """Metadata extraction should return complete patient info."""
        from src.dicom_handler import extract_metadata
        
        metadata = extract_metadata(sample_dicom_path)
        
        assert metadata["patient_name"] == "Test^Patient"
        assert metadata["modality"] == "CR"
        assert metadata["body_part"] == "CHEST"
        assert metadata["rows"] == "256"
        assert metadata["columns"] == "256"
    
    def test_dicom_to_png(self, sample_dicom_path, tmp_path):
        """DICOM to PNG conversion should produce valid image."""
        from src.dicom_handler import dicom_to_png
        
        output_path = str(tmp_path / "output.png")
        result = dicom_to_png(sample_dicom_path, output_path=output_path)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert os.path.exists(output_path)
    
    def test_anonymize_dicom(self, sample_dicom_path, tmp_path):
        """Anonymization should remove PHI tags."""
        from src.dicom_handler import anonymize_dicom, extract_metadata
        
        anon_path = str(tmp_path / "anonymized.dcm")
        removed_phi = anonymize_dicom(sample_dicom_path, output_path=anon_path)
        
        # Should have removed some PHI
        assert len(removed_phi) > 0
        assert "PatientName" in removed_phi
        assert removed_phi["PatientName"] == "Test^Patient"
        
        # Verify anonymized file
        anon_metadata = extract_metadata(anon_path)
        assert anon_metadata["patient_name"] == "ANONYMOUS"
        assert anon_metadata["patient_id"] == "000000"
    
    def test_anonymize_preserves_pixels(self, sample_dicom_path, tmp_path):
        """Anonymization should not modify pixel data."""
        from src.dicom_handler import load_dicom, anonymize_dicom
        
        # Load original
        original_pixels, _ = load_dicom(sample_dicom_path)
        
        # Anonymize
        anon_path = str(tmp_path / "anon.dcm")
        anonymize_dicom(sample_dicom_path, output_path=anon_path)
        
        # Load anonymized
        anon_pixels, _ = load_dicom(anon_path)
        
        assert np.array_equal(original_pixels, anon_pixels), \
            "Pixel data should be unchanged after anonymization"
    
    def test_anonymize_removes_private_tags(self, sample_dicom_path, tmp_path):
        """Private tags should be removed during anonymization."""
        from src.dicom_handler import anonymize_dicom
        
        anon_path = str(tmp_path / "anon.dcm")
        anonymize_dicom(sample_dicom_path, output_path=anon_path, remove_private_tags=True)
        
        ds = pydicom.dcmread(anon_path)
        
        # Check de-identification marker
        assert getattr(ds, 'PatientIdentityRemoved', '') == 'YES'

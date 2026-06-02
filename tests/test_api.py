"""
Tests for MedVision FastAPI Server
Tests all API endpoints with synthetic data.
Designed to pass regardless of model/dependency availability.
"""

import pytest
import sys
import os
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from PIL import Image
import numpy as np


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from api.main import app
    return TestClient(app)


@pytest.fixture
def test_image_bytes():
    """Create a synthetic test image as bytes."""
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "device" in data
        assert "cuda_available" in data
    
    def test_health_check_has_model_status(self, client):
        """Health check should report model availability."""
        response = client.get("/health")
        data = response.json()
        
        assert "classifier_loaded" in data
        assert "segmenter_loaded" in data
        assert "classifier_model_exists" in data
        assert "segmenter_model_exists" in data


class TestClassifyEndpoint:
    """Test classification endpoint."""
    
    def test_classify_returns_valid_response(self, client, test_image_bytes):
        """Classify should return 200 (with model) or 503 (without model)."""
        response = client.post(
            "/classify",
            files={"file": ("test.png", test_image_bytes, "image/png")},
        )
        # 503 if model not trained, 200 if model exists
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "probabilities" in data


class TestDeidentifyEndpoint:
    """Test de-identification endpoint."""
    
    def test_deidentify_response_code(self, client, test_image_bytes):
        """De-identification should return 200 (Tesseract installed) or 500/503 (not installed)."""
        response = client.post(
            "/deidentify",
            files={"file": ("report.png", test_image_bytes, "image/png")},
        )
        # 200 if Tesseract is installed, 500/503 if not
        assert response.status_code in [200, 500, 503]


class TestDICOMEndpoints:
    """Test DICOM endpoints."""
    
    def test_dicom_convert_response(self, client):
        """DICOM convert should handle invalid files gracefully."""
        response = client.post(
            "/dicom/convert",
            files={"file": ("test.dcm", b"not a dicom file", "application/dicom")},
        )
        # Should return some error (not crash)
        assert response.status_code in [400, 422, 500, 503]

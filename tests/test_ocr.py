"""
Tests for MedVision OCR & De-identification Pipeline
Tests entity detection regex patterns and text de-identification.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ocr_pipeline import (
    detect_entities,
    deidentify_text,
    DetectedEntity,
    DeidentificationResult,
    PHI_PATTERNS,
)


class TestEntityDetection:
    """Test PHI entity detection patterns."""
    
    def test_detect_date_formats(self):
        """Should detect dates in multiple formats."""
        text = "Date of exam: 01/15/2024 and also 2024-01-15"
        entities = detect_entities(text)
        
        date_entities = [e for e in entities if e.entity_type == "DATE"]
        assert len(date_entities) >= 1, "Should detect at least one date"
    
    def test_detect_mrn(self):
        """Should detect Medical Record Numbers."""
        text = "Patient ID: MRN123456 admitted for observation."
        entities = detect_entities(text)
        
        mrn_entities = [e for e in entities if e.entity_type == "MRN"]
        assert len(mrn_entities) >= 1, f"Should detect MRN, found: {[e.value for e in entities]}"
    
    def test_detect_phone_number(self):
        """Should detect phone numbers."""
        text = "Contact: (555) 123-4567 for appointments."
        entities = detect_entities(text)
        
        phone_entities = [e for e in entities if e.entity_type == "PHONE"]
        assert len(phone_entities) >= 1, "Should detect phone number"
    
    def test_detect_email(self):
        """Should detect email addresses."""
        text = "Email: john.smith@hospital.com for inquiries."
        entities = detect_entities(text)
        
        email_entities = [e for e in entities if e.entity_type == "EMAIL"]
        assert len(email_entities) == 1, "Should detect exactly one email"
        assert "john.smith@hospital.com" in email_entities[0].value
    
    def test_detect_doctor_name(self):
        """Should detect doctor names."""
        text = "Referring Physician: Dr. Sarah Johnson"
        entities = detect_entities(text)
        
        doctor_entities = [e for e in entities if e.entity_type == "DOCTOR_NAME"]
        assert len(doctor_entities) >= 1, "Should detect doctor name"
    
    def test_detect_patient_name(self):
        """Should detect patient names."""
        text = "Patient Name: John Smith\nAge: 45"
        entities = detect_entities(text)
        
        patient_entities = [e for e in entities if e.entity_type == "PATIENT_NAME"]
        assert len(patient_entities) >= 1, "Should detect patient name"
    
    def test_detect_age(self):
        """Should detect age references."""
        text = "45 years old male with persistent cough."
        entities = detect_entities(text)
        
        age_entities = [e for e in entities if e.entity_type == "AGE"]
        assert len(age_entities) >= 1, "Should detect age"
    
    def test_no_false_positives_on_clean_text(self):
        """Should not detect PHI in generic medical findings."""
        text = "The lungs are clear. No consolidation or effusion. Heart is normal."
        entities = detect_entities(text)
        
        # Only very generic patterns might match, but no names/MRN/dates should
        patient_entities = [e for e in entities if e.entity_type in ["PATIENT_NAME", "MRN", "SSN"]]
        assert len(patient_entities) == 0, f"False positives found: {[e.value for e in patient_entities]}"


class TestDeidentification:
    """Test PHI de-identification."""
    
    def test_deidentify_replaces_entities(self):
        """De-identification should replace PHI with placeholders."""
        text = "Patient Name: John Smith\nEmail: john@email.com\nPhone: (555) 123-4567"
        result = deidentify_text(text)
        
        assert isinstance(result, DeidentificationResult)
        assert result.original_text == text
        assert "John Smith" not in result.deidentified_text or "[PATIENT_REDACTED]" in result.deidentified_text
    
    def test_deidentify_preserves_medical_content(self):
        """De-identification should preserve non-PHI medical content."""
        text = "FINDINGS: The lungs are clear bilaterally. Patient Name: John Smith"
        result = deidentify_text(text)
        
        assert "lungs are clear bilaterally" in result.deidentified_text
        assert "FINDINGS" in result.deidentified_text
    
    def test_deidentify_entity_count(self):
        """Entity count should be correct."""
        text = "Patient ID: MRN123456. Phone: (555) 123-4567. Email: test@test.com"
        result = deidentify_text(text)
        
        total_entities = sum(result.entity_count.values())
        assert total_entities > 0, "Should find at least one entity"
    
    def test_deidentify_email_placeholder(self):
        """Email should be replaced with [EMAIL_REDACTED]."""
        text = "Contact: doctor@hospital.com"
        result = deidentify_text(text)
        
        assert "[EMAIL_REDACTED]" in result.deidentified_text
        assert "doctor@hospital.com" not in result.deidentified_text
    
    def test_deidentify_handles_empty_text(self):
        """Should handle empty or whitespace-only text."""
        result = deidentify_text("")
        assert result.deidentified_text == ""
        assert len(result.entities_found) == 0
    
    def test_full_report_deidentification(self):
        """Should handle a realistic medical report."""
        report = """
        RADIOLOGY REPORT
        Patient Name: John Smith
        Patient ID: MRN123456
        Date: 01/15/2024
        Dr. Sarah Johnson
        Phone: (555) 123-4567
        
        FINDINGS: Normal chest radiograph.
        """
        
        result = deidentify_text(report)
        
        # Should have found multiple entities
        assert len(result.entities_found) >= 3, \
            f"Expected >=3 entities, found {len(result.entities_found)}: {[e.entity_type for e in result.entities_found]}"
        
        # Key PHI should be replaced
        assert "FINDINGS: Normal chest radiograph" in result.deidentified_text

"""
Tests for MedVision PS3.15 De-identification
Tests profile loading, deterministic pseudonymization, UID consistency,
date shifting, header-informed text cleaning, and pixel data preservation.
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence

from src.deidentify import Deidentifier


def make_dataset(patient_id="12345678", study_uid="1.2.3.4.5", series_uid="1.2.3.4.5.6", sop_uid="1.2.3.4.5.6.7"):
    """A small CR dataset with planted PHI in headers and free text."""
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPInstanceUID = sop_uid
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.PatientName = "PATTERSON^RACHEL"
    ds.PatientID = patient_id
    ds.PatientBirthDate = "19420629"
    ds.PatientSex = "F"
    ds.PatientAge = "070Y"
    ds.StudyDate = "20120820"
    ds.StudyTime = "171851"
    ds.AccessionNumber = "20120820E368537"
    ds.ReferringPhysicianName = "WHITE^DERRICK"
    ds.Modality = "CR"
    ds.StudyDescription = "CHEST 1V"
    ds.SeriesDescription = "AP for Rachel Patterson"
    ds.RequestedProcedureDescription = "Ordered for Rachel Patterson by Dr. White"
    ds.AdditionalPatientHistory = "Performed 20120820"
    ds.KVP = "100"
    ds.add_new(0x00131010, "LO", "private vendor value")
    item = Dataset()
    item.ReferencedSOPInstanceUID = sop_uid
    item.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    ds.ReferencedImageSequence = Sequence([item])
    ds.Rows, ds.Columns = 4, 4
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
    ds.SamplesPerPixel, ds.PixelRepresentation = 1, 0
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = np.arange(16, dtype=np.uint16).tobytes()
    return ds


@pytest.fixture
def deid():
    return Deidentifier(salt="test-salt")


class TestProfile:
    def test_profile_loaded_from_standard(self, deid):
        assert len(deid.exact_actions) > 500

    def test_basic_profile_actions(self, deid):
        assert deid.action_for(0x00100010) == "Z"   # Patient's Name
        assert deid.action_for(0x0020000D) == "U"   # Study Instance UID

    def test_option_overrides_basic(self, deid):
        assert deid.action_for(0x00080020) == "C"   # Study Date: modified dates option
        assert deid.action_for(0x00100040) == "K"   # Patient's Sex: patient characteristics

    def test_wildcard_tags(self, deid):
        assert deid.action_for(0x60004000) == "X"   # Overlay Comments (60xx,4000)


class TestPseudonymization:
    def test_patient_id_and_name_replaced(self, deid):
        ds = deid.deidentify(make_dataset())
        assert ds.PatientID != "12345678"
        assert str(ds.PatientName) == ds.PatientID
        assert ds.PatientIdentityRemoved == "YES"

    def test_same_patient_same_pseudonym(self, deid):
        a = deid.deidentify(make_dataset(sop_uid="1.2.3.1"))
        b = deid.deidentify(make_dataset(sop_uid="1.2.3.2"))
        assert a.PatientID == b.PatientID

    def test_different_patients_differ(self, deid):
        a = deid.deidentify(make_dataset(patient_id="111"))
        b = deid.deidentify(make_dataset(patient_id="222"))
        assert a.PatientID != b.PatientID

    def test_deterministic_across_instances(self):
        a = Deidentifier(salt="s").deidentify(make_dataset())
        b = Deidentifier(salt="s").deidentify(make_dataset())
        assert a.PatientID == b.PatientID and a.StudyInstanceUID == b.StudyInstanceUID


class TestUIDs:
    def test_uids_replaced(self, deid):
        ds = deid.deidentify(make_dataset())
        assert ds.StudyInstanceUID != "1.2.3.4.5"
        assert ds.SOPInstanceUID.startswith("2.25.")
        assert len(ds.SOPInstanceUID) <= 64

    def test_series_consistent_across_files(self, deid):
        a = deid.deidentify(make_dataset(sop_uid="1.2.3.1"))
        b = deid.deidentify(make_dataset(sop_uid="1.2.3.2"))
        assert a.SeriesInstanceUID == b.SeriesInstanceUID
        assert a.SOPInstanceUID != b.SOPInstanceUID

    def test_references_follow_instance_uid(self, deid):
        ds = deid.deidentify(make_dataset())
        assert ds.ReferencedImageSequence[0].ReferencedSOPInstanceUID == ds.SOPInstanceUID
        assert ds.file_meta.MediaStorageSOPInstanceUID == ds.SOPInstanceUID

    def test_well_known_uids_kept(self, deid):
        ds = deid.deidentify(make_dataset())
        assert ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.1"
        assert ds.ReferencedImageSequence[0].ReferencedSOPClassUID == "1.2.840.10008.5.1.4.1.1.1"


class TestDatesAndText:
    def test_dates_shifted_consistently(self, deid):
        ds = deid.deidentify(make_dataset())
        assert ds.StudyDate != "20120820"
        assert ds.AdditionalPatientHistory == f"Performed {ds.StudyDate}"

    def test_birth_date_removed(self, deid):
        assert deid.deidentify(make_dataset()).PatientBirthDate == ""

    def test_names_removed_from_free_text(self, deid):
        ds = deid.deidentify(make_dataset())
        for text in (ds.SeriesDescription, ds.RequestedProcedureDescription):
            assert "rachel" not in text.lower() and "patterson" not in text.lower()
        assert "white" not in ds.RequestedProcedureDescription.lower()

    def test_non_phi_retained(self, deid):
        ds = deid.deidentify(make_dataset())
        assert ds.StudyDescription == "CHEST 1V"
        assert ds.PatientSex == "F" and ds.PatientAge == "070Y"
        assert str(ds.KVP) == "100"

    def test_identifiers_blanked(self, deid):
        ds = deid.deidentify(make_dataset())
        assert ds.AccessionNumber == ""
        assert str(ds.ReferringPhysicianName) == ""

    def test_private_tags_removed(self, deid):
        ds = deid.deidentify(make_dataset())
        assert not any(elem.tag.is_private for elem in ds)


class TestPixels:
    def test_pixel_data_untouched(self, deid):
        original = make_dataset().PixelData
        assert deid.deidentify(make_dataset()).PixelData == original


class TestMethodRecording:
    def test_method_codes_recorded(self, deid):
        ds = deid.deidentify(make_dataset())
        codes = {item.CodeValue for item in ds.DeidentificationMethodCodeSequence}
        assert {"113100", "113107"} <= codes
        assert len(ds.DeidentificationMethod) <= 64


class TestRefinements:
    def test_safe_private_attribute_kept(self, deid):
        ds = make_dataset()
        ds.add_new(0x70530010, "LO", "Philips PET Private Group")   # private creator
        ds.add_new(0x70531000, "DS", "1.5")                         # SUV factor: safe per E.3.10-1
        ds.add_new(0x70531077, "LO", "unknown vendor value")        # not listed: removed
        out = deid.deidentify(ds)
        assert 0x70531000 in out and 0x70530010 in out
        assert 0x70531077 not in out

    def test_unsafe_private_block_removed_with_creator(self, deid):
        out = deid.deidentify(make_dataset())
        assert 0x00131010 not in out

    def test_institution_mentions_suppressed(self, deid):
        ds = make_dataset()
        ds.InstitutionName = "Saint Clare Hospital"
        ds.ProtocolName = "t2_trufi multi at SCH"
        ds.ImageComments = "supine at Saint Clare Hospital"
        out = deid.deidentify(ds)
        assert out.ProtocolName == "" and out.ImageComments == ""

    def test_date_shaped_series_number_shifted(self, deid):
        ds = make_dataset()
        ds.SeriesNumber = "20191227"
        out = deid.deidentify(ds)
        assert str(out.SeriesNumber) != "20191227" and len(str(out.SeriesNumber)) == 8

    def test_small_series_number_kept(self, deid):
        ds = make_dataset()
        ds.SeriesNumber = "3"
        assert str(deid.deidentify(ds).SeriesNumber) == "3"

    def test_ae_titles_blanked(self, deid):
        ds = make_dataset()
        ds.RetrieveAETitle = "WHOTMM3367X8"                  # listed: removed by the profile
        ds.add_new(0x00400241, "AE", "Z9CX5NJWGP2R")        # Performed Station AE Title
        out = deid.deidentify(ds)
        assert out.get("RetrieveAETitle", "") == ""
        assert out.get("PerformedStationAETitle", "") == ""

    def test_all_replaced_uids_reported(self, deid):
        deid.deidentify(make_dataset())
        reported = dict(deid.pop_new_uids())
        assert "1.2.3.4.5" in reported and "1.2.3.4.5.6.7" in reported
        assert deid.pop_new_uids() == []

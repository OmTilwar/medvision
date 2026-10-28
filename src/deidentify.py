"""
MedVision DICOM De-identification (PS3.15 profile)

Applies the DICOM PS3.15 Basic Application Level Confidentiality Profile, with
selected profile options, using the action table generated from the standard
(src/ps315_profile.json, see scripts/build_ps315_profile.py).

Design:
  - Deterministic pseudonymization: replacement UIDs, patient pseudonyms and
    per-patient date offsets are derived from a keyed hash of the original
    value, so the same study/series/patient always maps to the same output
    across files and parallel workers, without shared state.
  - Dates are shifted by a per-patient offset (Retain Longitudinal Temporal
    Information with Modified Dates option), preserving intervals between scans.
  - Header-informed text cleaning: identifying values found in the file's own
    header (names, IDs, accession numbers, dates) are removed or shifted
    wherever they reappear in free-text attributes, plus regex PHI patterns.
  - Pixel data is never modified unless burned-in text redaction is enabled
    (see redact_burned_in_text).
"""

import os
import re
import json
import hashlib
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from src.ocr_pipeline import detect_entities

PROFILE_PATH = os.path.join(os.path.dirname(__file__), "ps315_profile.json")

# Profile options applied on top of the Basic Profile (PS3.15 E.3)
DEFAULT_OPTIONS = (
    "retain_safe_private",
    "retain_longitudinal_modified_dates",
    "retain_patient_characteristics",
    "clean_descriptors",
    "clean_structured_content",
)

# Combined action codes resolved to the least destructive conforming action,
# so attributes that may be Type 1/2 in the IOD are kept present
COMBINED_ACTIONS = {"X/Z": "Z", "X/D": "D", "X/Z/D": "D", "Z/D": "D", "X/Z/U*": "U"}

# CID 7050 De-identification Method codes for the profile and options used
METHOD_CODES = {
    "basic": ("113100", "Basic Application Confidentiality Profile"),
    "clean_structured_content": ("113104", "Clean Structured Content Option"),
    "clean_descriptors": ("113105", "Clean Descriptors Option"),
    "retain_longitudinal_modified_dates": ("113107", "Retain Longitudinal Temporal Information Modified Dates Option"),
    "retain_patient_characteristics": ("113108", "Retain Patient Characteristics Option"),
    "retain_device_identity": ("113109", "Retain Device Identity Option"),
    "retain_safe_private": ("113111", "Retain Safe Private Option"),
}

# Words too generic to scrub on their own when they appear in an institution name
GENERIC_INSTITUTION_WORDS = {
    "hospital", "medical", "center", "centre", "clinic", "university", "health",
    "imaging", "radiology", "department", "the", "of", "and", "for", "at", "general",
}

WELL_KNOWN_UID_ROOT = "1.2.840.10008."
TEXT_VRS = {"LO", "SH", "ST", "LT", "UT", "UC", "PN", "CS", "AE"}
# Institution Name, Institution Address, Station Name, Institutional Department Name
INSTITUTION_TAGS = {0x00080080, 0x00080081, 0x00081010, 0x00081040}
FREE_TEXT_VRS = {"LO", "SH", "ST", "LT", "UT", "UC"}

DATE_PATTERNS = [
    (re.compile(r"\b(\d{4})(\d{2})(\d{2})\b"), "%Y%m%d"),
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "%Y-%m-%d"),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), "%m/%d/%Y"),
]


def _hash_int(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest(), 16)


class Deidentifier:
    """De-identify pydicom datasets with the PS3.15 profile."""

    def __init__(
        self,
        salt: str,
        options: Iterable[str] = DEFAULT_OPTIONS,
        profile_path: str = PROFILE_PATH,
        max_date_shift_days: int = 3650,
    ):
        self.salt = salt
        self.options = tuple(options)
        self.max_date_shift_days = max_date_shift_days
        self.exact_actions, self.masked_actions = self._load_profile(profile_path)
        self.safe_private = self._load_safe_private(profile_path)
        # old -> new, collected for mapping files and audit
        self.uid_map: Dict[str, str] = {}
        self.patient_map: Dict[str, str] = {}
        self._fresh_uids: List[Tuple[str, str]] = []

    def pop_new_uids(self) -> List[Tuple[str, str]]:
        """UID replacements created since the last call (for streaming mapping files)."""
        fresh, self._fresh_uids = self._fresh_uids, []
        return fresh

    # ── Profile ────────────────────────────────────────────────

    def _load_profile(self, path):
        with open(path, encoding="utf-8") as f:
            attributes = json.load(f)["attributes"]
        exact, masked = {}, []
        for attr in attributes:
            action = self._resolve_action(attr["actions"])
            if action is None:
                continue
            tag = attr["tag"].replace(",", "")
            if not re.fullmatch(r"[0-9A-FX]{8}", tag):
                continue   # descriptive rows, e.g. "private attributes (gggg,eeee) where gggg is odd"
            if "X" in tag:
                pattern = re.compile("^" + tag.replace("X", "[0-9A-F]") + "$")
                masked.append((pattern, action))
            else:
                exact[int(tag, 16)] = action
        return exact, masked

    def _load_safe_private(self, path) -> Set[Tuple[int, str, int]]:
        """(group, private creator, element low byte) from PS3.15 Table E.3.10-1."""
        if "retain_safe_private" not in self.options:
            return set()
        with open(path, encoding="utf-8") as f:
            entries = json.load(f).get("safe_private", [])
        return {(int(e["group"], 16), e["creator"].strip().lower(), int(e["element"], 16))
                for e in entries if re.fullmatch(r"[0-9A-F]{4}", e["group"])}

    def _remove_unsafe_private(self, ds: Dataset):
        """Remove private attributes except those listed as safe in PS3.15 E.3.10-1."""
        creators = {}
        for elem in ds:
            if elem.tag.is_private and 0x10 <= elem.tag.element <= 0xFF:
                creators[(elem.tag.group, elem.tag.element)] = str(elem.value).strip().lower()

        kept_blocks = set()
        for elem in list(ds):
            tag = elem.tag
            if not tag.is_private or tag.element <= 0xFF:
                continue
            block = tag.element >> 8
            creator = creators.get((tag.group, block), "")
            if (tag.group, creator, tag.element & 0xFF) in self.safe_private:
                kept_blocks.add((tag.group, block))
            else:
                del ds[tag]
        for elem in list(ds):                 # creators whose block is now empty
            tag = elem.tag
            if tag.is_private and tag.element <= 0xFF and (tag.group, tag.element) not in kept_blocks:
                del ds[tag]

    def _resolve_action(self, actions: Dict[str, str]) -> Optional[str]:
        action = actions.get("basic")
        for option in self.options:          # options override the Basic Profile
            if actions.get(option):
                action = actions[option]
        if action is None:
            return None
        return COMBINED_ACTIONS.get(action, action)

    def action_for(self, tag) -> Optional[str]:
        tag = int(tag)
        if tag in self.exact_actions:
            return self.exact_actions[tag]
        hex_tag = f"{tag:08X}"
        for pattern, action in self.masked_actions:
            if pattern.match(hex_tag):
                return action
        return None

    # ── Deterministic replacements ─────────────────────────────

    def new_uid(self, uid: str) -> str:
        if uid not in self.uid_map:
            self.uid_map[uid] = f"2.25.{_hash_int(self.salt, 'uid', uid) % 10**38}"
            self._fresh_uids.append((uid, self.uid_map[uid]))
        return self.uid_map[uid]

    def pseudonym(self, patient_id: str) -> str:
        if patient_id not in self.patient_map:
            self.patient_map[patient_id] = f"MV{_hash_int(self.salt, 'patient', patient_id) % 10**8:08d}"
        return self.patient_map[patient_id]

    def date_offset(self, patient_id: str) -> timedelta:
        days = 1 + _hash_int(self.salt, "date", patient_id) % self.max_date_shift_days
        return timedelta(days=-days)

    # ── Main entry point ───────────────────────────────────────

    def deidentify(self, ds: Dataset) -> Dataset:
        """De-identify a dataset in place and return it."""
        patient_id = str(ds.get("PatientID", "") or "UNKNOWN")
        offset = self.date_offset(patient_id)
        phi_tokens = self._collect_phi_tokens(ds)

        self._process(ds, offset, phi_tokens)

        pseudonym = self.pseudonym(patient_id)
        ds.PatientID = pseudonym
        ds.PatientName = pseudonym
        ds.PatientIdentityRemoved = "YES"
        ds.DeidentificationMethod = "MedVision PS3.15 Basic Profile with options"
        ds.DeidentificationMethodCodeSequence = self._method_codes()

        if hasattr(ds, "file_meta") and "MediaStorageSOPInstanceUID" in ds.file_meta:
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        return ds

    def _method_codes(self) -> Sequence:
        """De-identification Method Code Sequence (CID 7050) for the profile and options."""
        items = []
        for key in ("basic",) + self.options:
            if key not in METHOD_CODES:
                continue
            item = Dataset()
            item.CodeValue, item.CodeMeaning = METHOD_CODES[key]
            item.CodingSchemeDesignator = "DCM"
            items.append(item)
        return Sequence(items)

    def _process(self, ds: Dataset, offset: timedelta, phi_tokens: Set[str]):
        self._remove_unsafe_private(ds)     # Basic Profile + Retain Safe Private option
        for elem in list(ds):
            tag = elem.tag
            if tag.is_private:
                continue                    # safe private attributes are kept as-is
            if tag == 0x7FE00010:           # Pixel Data is handled separately
                continue

            action = self.action_for(tag)

            if elem.VR == "SQ":
                if action == "X":
                    del ds[tag]
                    continue
                for item in elem.value:     # K/C/Z/D on sequences: clean contents
                    self._process(item, offset, phi_tokens)
                continue

            if action == "X":
                del ds[tag]
            elif action == "Z":
                elem.value = self._zero_or_dummy(elem)
            elif action == "D":
                elem.value = self._dummy(elem, offset)
            elif action == "U":
                elem.value = self._map_uids(elem.value)
            elif action == "C":
                elem.value = self._clean(elem, offset, phi_tokens)
            elif elem.VR == "UI":
                elem.value = self._map_uids(elem.value)   # unlisted UIDs are still remapped
            elif elem.VR == "AE":
                elem.value = ""                           # AE titles identify devices and sites
            elif elem.VR == "IS":
                elem.value = self._shift_date_shaped_number(elem.value, offset)
            elif elem.VR in FREE_TEXT_VRS or elem.VR == "PN":
                # Unlisted text: retained, but scrubbed of this header's PHI
                elem.value = self._clean_text(elem.value, offset, phi_tokens)

    # ── Actions ────────────────────────────────────────────────

    def _map_uids(self, value):
        if value is None or value == "":
            return value
        values = value if isinstance(value, (list, pydicom.multival.MultiValue)) else [value]
        mapped = [v if str(v).startswith(WELL_KNOWN_UID_ROOT) else self.new_uid(str(v)) for v in values]
        return mapped if len(mapped) > 1 else mapped[0]

    def _zero_or_dummy(self, elem):
        return "" if elem.VR in TEXT_VRS | {"DA", "DT", "TM", "AS", "UI"} else None

    def _dummy(self, elem, offset):
        vr = elem.VR
        if vr in ("DA", "DT"):
            return self._shift_date_value(elem.value, offset) or ""
        if vr == "TM":
            return elem.value                # times kept (temporal information retained)
        if vr == "UI":
            return self._map_uids(elem.value)
        if vr in ("IS", "DS", "US", "SS", "UL", "SL", "FL", "FD"):
            return 0
        if vr == "PN":
            return "ANONYMIZED"
        if vr in TEXT_VRS:
            return "ANONYMIZED"
        return elem.value

    def _clean(self, elem, offset, phi_tokens):
        if elem.VR in ("DA", "DT"):
            return self._shift_date_value(elem.value, offset)
        if elem.VR == "TM":
            return elem.value
        if elem.VR in TEXT_VRS:
            return self._clean_text(elem.value, offset, phi_tokens)
        return elem.value

    # ── Dates ──────────────────────────────────────────────────

    @staticmethod
    def _shift_date_string(text: str, offset: timedelta) -> Optional[str]:
        """Shift a DA (YYYYMMDD) or DT (YYYYMMDD...) string; None if unparseable."""
        if len(text) < 8 or not text[:8].isdigit():
            return None
        try:
            shifted = datetime.strptime(text[:8], "%Y%m%d").date() + offset
        except ValueError:
            return None
        return shifted.strftime("%Y%m%d") + text[8:]

    def _shift_date_value(self, value, offset):
        if value in (None, ""):
            return value
        values = value if isinstance(value, (list, pydicom.multival.MultiValue)) else [value]
        shifted = [self._shift_date_string(str(v), offset) or "" for v in values]
        return shifted if len(shifted) > 1 else shifted[0]

    def _shift_date_shaped_number(self, value, offset):
        """Integers that are really dates (e.g. Series Number 20191227) are shifted too."""
        if value in (None, "") or isinstance(value, (list, pydicom.multival.MultiValue)):
            return value
        text = str(value)
        if len(text) == 8 and text.isdigit() and 1900 <= int(text[:4]) <= 2100:
            shifted = self._shift_date_string(text, offset)
            if shifted:
                return shifted
        return value

    def _shift_dates_in_text(self, text: str, offset: timedelta) -> str:
        for pattern, fmt in DATE_PATTERNS:
            def repl(match, fmt=fmt):
                try:
                    parsed = datetime.strptime(match.group(0), fmt).date()
                except ValueError:
                    return match.group(0)
                return (parsed + offset).strftime(fmt)
            text = pattern.sub(repl, text)
        return text

    # ── Text cleaning ──────────────────────────────────────────

    def _collect_phi_tokens(self, ds: Dataset) -> Set[str]:
        """Identifying strings from this file's own header (names, IDs, numbers)."""
        tokens: Set[str] = set()

        def visit(dataset):
            for elem in dataset:
                if elem.VR == "SQ":
                    for item in elem.value:
                        visit(item)
                    continue
                if elem.value in (None, ""):
                    continue
                if elem.VR == "PN":
                    for part in re.split(r"[\^=\s,]+", str(elem.value)):
                        if len(part) >= 2:
                            tokens.add(part.lower())
                elif elem.tag in INSTITUTION_TAGS:
                    tokens.update(self._institution_tokens(str(elem.value)))
                elif self.action_for(elem.tag) in ("Z", "D", "X") and elem.VR in ("LO", "SH") \
                        and elem.tag != 0x00080060:
                    value = str(elem.value).strip()
                    if len(value) >= 4 and any(ch.isdigit() for ch in value):
                        tokens.add(value.lower())   # identifiers: IDs, accession numbers
        visit(ds)
        return tokens

    @staticmethod
    def _institution_tokens(name: str) -> Set[str]:
        """Full institution/station name, its acronym, and its distinctive words."""
        tokens = set()
        name = name.strip()
        if len(name) < 2:
            return tokens
        tokens.add(name.lower())
        words = [w for w in re.split(r"[\s,\-/]+", name) if w]
        if len(words) >= 2:
            acronym = "".join(w[0] for w in words if w[0].isalpha())
            if len(acronym) >= 2:
                tokens.add(acronym.lower())
        for word in words:
            if len(word) >= 4 and word.lower() not in GENERIC_INSTITUTION_WORDS:
                tokens.add(word.lower())
        return tokens

    def _clean_text(self, value, offset, phi_tokens):
        if value in (None, ""):
            return value
        values = value if isinstance(value, (list, pydicom.multival.MultiValue)) else [value]
        cleaned = [self._clean_string(str(v), offset, phi_tokens) for v in values]
        return cleaned if len(cleaned) > 1 else cleaned[0]

    def _clean_string(self, text: str, offset: timedelta, phi_tokens: Set[str]) -> str:
        """
        Shift embedded dates; if any identifying value is found, suppress the
        whole field — partial redaction leaves context ("Ordered for ... by Dr.")
        and residual fragments that regex cleaning cannot vouch for.
        """
        text = self._shift_dates_in_text(text, offset)
        for token in phi_tokens:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", text, flags=re.I):
                return ""
        if any(e.entity_type in ("SSN", "PHONE", "EMAIL", "MRN") for e in detect_entities(text)):
            return ""
        return text


# ──────────────────────────────────────────────
# Burned-in text redaction (optional)
# ──────────────────────────────────────────────

# Standard radiographic annotations that are not PHI and should stay visible
SAFE_ANNOTATIONS = {
    "r", "l", "pa", "ap", "lat", "erect", "supine", "upright", "portable", "port",
    "standing", "sitting", "decub", "lld", "rld", "cc", "mlo", "rcc", "lcc", "rmlo", "lmlo",
}


def redact_burned_in_text(ds: Dataset, reader, min_confidence: float = 0.3) -> int:
    """
    Black out burned-in text regions found by OCR, except standard annotations.

    Only single-frame grayscale images are handled. Returns the number of boxes
    redacted; pixel data is rewritten (uncompressed) only if something is found.
    """
    if int(ds.get("NumberOfFrames", 1) or 1) != 1 or ds.get("SamplesPerPixel", 1) != 1:
        return 0
    pixels = ds.pixel_array
    lo, hi = float(pixels.min()), float(pixels.max())
    if hi <= lo:
        return 0
    image8 = ((pixels.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)

    boxes = []
    for bbox, text, conf in reader.readtext(image8):
        word = re.sub(r"[^a-z0-9]", "", text.lower())
        if conf < min_confidence or len(word) < 2 or word in SAFE_ANNOTATIONS:
            continue
        xs = [int(p[0]) for p in bbox]
        ys = [int(p[1]) for p in bbox]
        boxes.append((max(min(xs) - 2, 0), max(min(ys) - 2, 0), max(xs) + 2, max(ys) + 2))

    if not boxes:
        return 0
    fill = pixels.min()
    redacted = pixels.copy()
    for x0, y0, x1, y1 in boxes:
        redacted[y0:y1, x0:x1] = fill

    if ds.file_meta.TransferSyntaxUID.is_compressed:
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.PixelData = redacted.tobytes()
    ds.BurnedInAnnotation = "NO"
    return len(boxes)

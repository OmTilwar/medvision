"""
Batch-de-identify a directory of DICOM files and write the UID / patient-ID
mapping CSVs used by the MIDI-B validation script.

Methods:
  legacy  - the original src/dicom_handler.anonymize_dicom (tag list + random UIDs)
  ps315   - src/deidentify.Deidentifier (PS3.15 Basic Profile + options)

Usage:
    python scripts/deidentify_dataset.py --input C:/datasets/midi-b/synthetic-validation \\
        --output C:/datasets/midi-b/runs/ps315 --method ps315 [--redact-pixels]
"""

import os
import sys
import csv
import argparse
import contextlib
import io
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pydicom

UID_KEYWORDS = ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "FrameOfReferenceUID")
PIXEL_MODALITIES = {"CR", "DX", "MG", "US", "SC", "OT", "XA", "RF"}

_worker = {}


def _init_worker(method, salt, redact_pixels):
    _worker["method"] = method
    if method == "ps315":
        from src.deidentify import Deidentifier
        _worker["deid"] = Deidentifier(salt=salt)
    if redact_pixels:
        import easyocr
        _worker["ocr"] = easyocr.Reader(["en"], gpu=True, verbose=False)


def _process_file(job):
    src, out_dir = job
    try:
        original = pydicom.dcmread(src, stop_before_pixels=True)
        old = {k: str(original.get(k, "")) for k in UID_KEYWORDS + ("PatientID", "Modality")}

        if _worker["method"] == "legacy":
            from src.dicom_handler import anonymize_dicom
            tmp = os.path.join(out_dir, f"tmp_{os.getpid()}_{os.path.basename(src)}")
            with contextlib.redirect_stdout(io.StringIO()):
                anonymize_dicom(src, output_path=tmp)
            ds = pydicom.dcmread(tmp, stop_before_pixels=True)
            new_path = os.path.join(out_dir, ds.SeriesInstanceUID, f"{ds.SOPInstanceUID}.dcm")
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            os.replace(tmp, new_path)
        else:
            ds = pydicom.dcmread(src)
            boxes = 0
            if "ocr" in _worker and old["Modality"] in PIXEL_MODALITIES:
                from src.deidentify import redact_burned_in_text
                boxes = redact_burned_in_text(ds, _worker["ocr"])
            _worker["deid"].deidentify(ds)
            new_path = os.path.join(out_dir, ds.SeriesInstanceUID, f"{ds.SOPInstanceUID}.dcm")
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            ds.save_as(new_path, enforce_file_format=True)
            old["boxes"] = boxes

        new = {k: str(ds.get(k, "")) for k in UID_KEYWORDS + ("PatientID",)}
        # Every UID replaced in this file (references, creator UIDs, ...), not just the main four
        extra = _worker["deid"].pop_new_uids() if "deid" in _worker else []
        return {"ok": True, "old": old, "new": new, "boxes": old.get("boxes", 0), "uids": extra}
    except Exception as exc:
        return {"ok": False, "src": src, "error": repr(exc)}


def write_mapping(path, pairs):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id_old", "id_new"])
        for old, new in pairs:
            writer.writerow([old, new])


def main():
    parser = argparse.ArgumentParser(description="Batch DICOM de-identification")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=["legacy", "ps315"], default="ps315")
    parser.add_argument("--salt", default="medvision-midi-b")
    parser.add_argument("--redact-pixels", action="store_true", help="OCR-based burned-in text redaction")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    files = [os.path.join(root, f) for root, _, names in os.walk(args.input)
             for f in names if f.lower().endswith(".dcm")]
    data_dir = os.path.join(args.output, "data")
    os.makedirs(data_dir, exist_ok=True)
    workers = 1 if args.redact_pixels else args.workers   # one GPU OCR model
    print(f"{len(files)} files, method={args.method}, workers={workers}, redact_pixels={args.redact_pixels}")

    # Fail fast on configuration errors: a Pool whose initializer raises keeps
    # respawning workers and never finishes
    _init_worker(args.method, args.salt, False)

    start = time.time()
    uid_pairs, patient_pairs, failures, boxes = {}, {}, [], 0
    with Pool(workers, initializer=_init_worker, initargs=(args.method, args.salt, args.redact_pixels)) as pool:
        for i, result in enumerate(pool.imap_unordered(_process_file, [(f, data_dir) for f in files], chunksize=16), 1):
            if not result["ok"]:
                failures.append(result)
                continue
            boxes += result["boxes"]
            for key in UID_KEYWORDS:
                old, new = result["old"][key], result["new"][key]
                if old and old not in uid_pairs:      # first mapping wins, as in the scorer
                    uid_pairs[old] = new
            for old, new in result["uids"]:
                uid_pairs.setdefault(old, new)
            pid = result["old"]["PatientID"]
            if pid not in patient_pairs:
                patient_pairs[pid] = result["new"]["PatientID"]
            if i % 2000 == 0:
                print(f"  {i}/{len(files)} ({time.time() - start:.0f}s)", flush=True)

    write_mapping(os.path.join(args.output, "uid_mapping.csv"), uid_pairs.items())
    write_mapping(os.path.join(args.output, "patid_mapping.csv"), patient_pairs.items())
    print(f"Done in {time.time() - start:.0f}s: {len(files) - len(failures)} ok, {len(failures)} failed, "
          f"{len(uid_pairs)} UIDs, {len(patient_pairs)} patients, {boxes} text boxes redacted")
    for failure in failures[:10]:
        print(f"  FAILED {failure['src']}: {failure['error']}")


if __name__ == "__main__":
    main()

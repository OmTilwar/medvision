"""
Download a MIDI-B (Medical Image De-Identification Benchmark) image set from TCIA.

MIDI-B synthetic sets are DICOM studies with planted, fake PHI in headers and
burned into pixels; the matching answer key (Aspera download from TCIA) lists
every insertion and the expected de-identification action.
Collection page: https://doi.org/10.7937/cf2p-aw56 (CC BY 4.0)

Reads a TCIA .tcia manifest and fetches each series through the public NBIA
API, extracting it to <out-dir>/<SeriesInstanceUID>/. Already-downloaded series
are skipped, so the script can be re-run after an interruption.

Usage:
    python scripts/download_midi_b.py --manifest TCIA-MIDI-B-Synthetic-Validation_20250502.tcia \\
        --out-dir C:/datasets/midi-b/synthetic-validation
"""

import os
import sys
import argparse
import zipfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={}"


def read_manifest(path):
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    start = lines.index("ListOfSeriesToDownload=") + 1
    return [uid for uid in lines[start:] if uid]


def fetch_series(uid, out_dir, retries=4):
    series_dir = os.path.join(out_dir, uid)
    done_marker = os.path.join(series_dir, ".complete")
    if os.path.exists(done_marker):
        return uid, "skipped"

    zip_path = os.path.join(out_dir, f"{uid}.zip.part")
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(API_URL.format(uid), zip_path)
            os.makedirs(series_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                members = [m for m in zf.namelist() if m.lower().endswith(".dcm")]
                for member in members:
                    # Flatten and sanitize member names; never trust archive paths
                    target = os.path.join(series_dir, os.path.basename(member))
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
            os.remove(zip_path)
            open(done_marker, "w").close()
            return uid, f"{len(members)} files"
        except Exception as exc:  # network errors, truncated zips
            if attempt == retries:
                return uid, f"FAILED: {exc}"


def main():
    parser = argparse.ArgumentParser(description="Download a MIDI-B image set from a TCIA manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    series = read_manifest(args.manifest)
    print(f"{len(series)} series in manifest")

    failed = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch_series, uid, args.out_dir) for uid in series]
        for i, future in enumerate(as_completed(futures), 1):
            uid, status = future.result()
            if status.startswith("FAILED"):
                failed.append(uid)
            if i % 20 == 0 or status.startswith("FAILED"):
                print(f"  [{i}/{len(series)}] {uid}: {status}", flush=True)

    print(f"Done: {len(series) - len(failed)} series ok, {len(failed)} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

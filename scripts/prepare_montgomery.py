"""
Download and prepare the NLM Montgomery County chest X-ray set for lung segmentation.

138 frontal chest X-rays (80 normal, 58 TB) with manual left/right lung masks.
Source: https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/

Preparation:
  - 12-bit PNGs are rescaled to 8-bit per image (Image.convert("L") would clip them)
  - left and right masks are merged into one binary lung mask
  - images and masks are resized once to --size x --size so training does not
    decode ~4000x4900 px files every epoch

Output (one image per patient, same filename in both folders):
    data/montgomery/images/MCUCXR_XXXX_Y.png
    data/montgomery/masks/MCUCXR_XXXX_Y.png

Usage:
    python scripts/prepare_montgomery.py
"""

import os
import re
import sys
import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

BASE_URL = ("https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/"
            "Montgomery-County-CXR-Set/MontgomerySet")
SUBDIRS = {"image": "CXR_png", "left": "ManualMask/leftMask", "right": "ManualMask/rightMask"}
FILENAME_RE = re.compile(r"MCUCXR_\d{4}_[01]\.png")


def list_files(subdir):
    with urllib.request.urlopen(f"{BASE_URL}/{subdir}/index.html", timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    return sorted(set(FILENAME_RE.findall(html)))


def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)


def to_uint8(image):
    """Rescale a 8/12/16-bit grayscale image to 8-bit using its own range."""
    arr = np.asarray(image).astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).round().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Prepare the Montgomery lung segmentation set")
    parser.add_argument("--raw-dir", default=os.path.join(config.DATA_DIR, "montgomery_raw"))
    parser.add_argument("--out-dir", default=os.path.join(config.DATA_DIR, "montgomery"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    names = {kind: list_files(sub) for kind, sub in SUBDIRS.items()}
    common = sorted(set(names["image"]) & set(names["left"]) & set(names["right"]))
    print(f"Found {len(common)} images with both lung masks")

    jobs = []
    for kind, sub in SUBDIRS.items():
        os.makedirs(os.path.join(args.raw_dir, kind), exist_ok=True)
        for name in common:
            jobs.append((f"{BASE_URL}/{sub}/{name}", os.path.join(args.raw_dir, kind, name)))
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda job: download(*job), jobs))
    print(f"Downloaded {len(jobs)} files to {args.raw_dir}")

    img_out = os.path.join(args.out_dir, "images")
    mask_out = os.path.join(args.out_dir, "masks")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    for name in common:
        image = to_uint8(Image.open(os.path.join(args.raw_dir, "image", name)))
        left = np.asarray(Image.open(os.path.join(args.raw_dir, "left", name)).convert("L")) > 0
        right = np.asarray(Image.open(os.path.join(args.raw_dir, "right", name)).convert("L")) > 0
        mask = ((left | right) * 255).astype(np.uint8)

        size = (args.size, args.size)
        Image.fromarray(image).resize(size, Image.BILINEAR).save(os.path.join(img_out, name))
        Image.fromarray(mask).resize(size, Image.NEAREST).save(os.path.join(mask_out, name))

    print(f"Prepared {len(common)} image/mask pairs at {args.size}x{args.size} in {args.out_dir}")


if __name__ == "__main__":
    main()

"""
5-fold cross-validation of the U-Net lung segmenter on the Montgomery County set.

Montgomery has one X-ray per patient (138 total), so a K-fold split over images
is patient-level. Each fold trains for a fixed number of epochs and is scored
once at the end, so held-out images never influence checkpoint selection.
Every image is scored exactly once by a model that never saw it, giving 138
out-of-fold Dice/IoU scores.

After cross-validation, a final model is trained on all images and saved to
outputs/models/segmenter_best.pth for the API.

Usage:
    python scripts/prepare_montgomery.py        # once
    python scripts/cross_validate_segmenter.py
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.dataset import SegmentationDataset
from src.segmenter import build_segmenter, BCEDiceLoss
from src.train_segmenter import train_one_epoch
from src.evaluate import compute_dice_score, compute_iou, plot_segmentation_results

RESULTS_DIR = os.path.join(config.OUTPUT_DIR, "results")


def train_model(train_ds, args, device, desc):
    model = build_segmenter(device)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    for epoch in range(args.epochs):
        metrics = train_one_epoch(model, loader, criterion, optimizer, device, epoch)
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            print(f"  [{desc}] epoch {epoch + 1}/{args.epochs}  loss {metrics['loss']:.4f}  train dice {metrics['dice']:.4f}")
    return model


@torch.no_grad()
def score(model, eval_ds, names, device):
    """Per-image Dice/IoU on held-out images; also returns arrays for plotting."""
    model.eval()
    rows, samples = [], []
    for i in range(len(eval_ds)):
        image, mask = eval_ds[i]
        pred = (torch.sigmoid(model(image.unsqueeze(0).to(device))) > 0.5).float().cpu().numpy().squeeze()
        true = mask.numpy().squeeze()
        rows.append({"image": names[i], "dice": compute_dice_score(pred, true), "iou": compute_iou(pred, true)})
        samples.append((image.numpy().squeeze(), true, pred))
    return rows, samples


def summarize(values):
    values = np.asarray(values)
    return {"mean": float(values.mean()), "std": float(values.std()),
            "median": float(np.median(values)), "min": float(values.min())}


def main():
    parser = argparse.ArgumentParser(description="5-fold CV of the U-Net on Montgomery County")
    parser.add_argument("--data-dir", default=os.path.join(config.DATA_DIR, "montgomery"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=config.LR_HEAD)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-final", action="store_true", help="Do not train the final all-data model")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = config.DEVICE

    train_view = SegmentationDataset(args.data_dir, is_train=True)
    eval_view = SegmentationDataset(args.data_dir, is_train=False)
    names = [os.path.basename(p) for p in eval_view.image_paths]
    n = len(eval_view)
    assert n > 0, f"No image/mask pairs in {args.data_dir} — run scripts/prepare_montgomery.py first"
    print(f"{n} image/mask pairs, {args.folds}-fold CV, {args.epochs} epochs/fold, device {device}")

    all_rows, fold_summaries, example_samples = [], [], []
    start = time.time()
    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (train_idx, test_idx) in enumerate(kfold.split(np.arange(n))):
        model = train_model(Subset(train_view, train_idx), args, device, f"fold {fold + 1}")
        rows, samples = score(model, Subset(eval_view, test_idx), [names[i] for i in test_idx], device)
        for row in rows:
            row["fold"] = fold + 1
        all_rows.extend(rows)
        if fold == 0:
            example_samples = samples[:6]
        summary = {"fold": fold + 1, "n_test": len(rows),
                   "dice_mean": float(np.mean([r["dice"] for r in rows])),
                   "iou_mean": float(np.mean([r["iou"] for r in rows]))}
        fold_summaries.append(summary)
        print(f"Fold {fold + 1}: n={summary['n_test']}  Dice {summary['dice_mean']:.4f}  IoU {summary['iou_mean']:.4f}")
        del model
        torch.cuda.empty_cache()

    results = {
        "dataset": "NLM Montgomery County CXR set (one image per patient)",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": f"{args.folds}-fold cross-validation, fixed {args.epochs} epochs per fold, "
                    "no checkpoint selection on held-out folds",
        "image_size": config.SEG_IMAGE_SIZE,
        "n_images": n,
        "hyperparameters": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                            "loss": "0.5 BCE + 0.5 Dice", "scheduler": "cosine", "seed": args.seed},
        "dice": summarize([r["dice"] for r in all_rows]),
        "iou": summarize([r["iou"] for r in all_rows]),
        "folds": fold_summaries,
        "per_image": sorted(all_rows, key=lambda r: r["image"]),
        "cv_minutes": round((time.time() - start) / 60, 1),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "segmenter_cv_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    images, trues, preds = zip(*example_samples)
    plot_segmentation_results(
        list(images), list(trues), list(preds),
        dice_scores=[compute_dice_score(p, t) for t, p in zip(trues, preds)],
        save_path=os.path.join(config.VIZ_DIR, "montgomery_cv_examples.png"),
    )

    print(f"\nOut-of-fold Dice {results['dice']['mean']:.4f} ± {results['dice']['std']:.4f}  "
          f"IoU {results['iou']['mean']:.4f} ± {results['iou']['std']:.4f}  (n={n})")
    print(f"Saved {out_path}")

    if not args.skip_final:
        model = train_model(train_view, args, device, "final")
        torch.save({"model_state_dict": model.state_dict(), "trained_on": results["dataset"],
                    "epochs": args.epochs}, config.SEGMENTER_BEST_PATH)
        print(f"Saved final all-data model to {config.SEGMENTER_BEST_PATH}")


if __name__ == "__main__":
    main()

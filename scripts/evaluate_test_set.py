"""
Evaluate the trained chest X-ray classifier on the held-out test split and
save the results, so every reported metric has a reproducible artifact.

The test split (data/chest_xray/test, 624 images) is never used for training
or checkpoint selection: classifier_best.pth is chosen on the validation split.

Usage:
    python scripts/evaluate_test_set.py
    python scripts/evaluate_test_set.py --checkpoint outputs/models/classifier_last.pth
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.classifier import load_classifier
from src.dataset import ChestXrayDataset
from src.preprocessing import get_classification_test_transforms
from src.evaluate import (
    compute_classification_metrics,
    compute_sensitivity_specificity,
    plot_confusion_matrix,
    plot_roc_curves,
)

RESULTS_DIR = os.path.join(config.OUTPUT_DIR, "results")


@torch.no_grad()
def predict(model, loader, device):
    all_probs, all_targets = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_targets.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_targets)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the classifier on the held-out test split")
    parser.add_argument("--checkpoint", default=config.CLASSIFIER_BEST_PATH)
    parser.add_argument("--test-dir", default=os.path.join(config.CHEST_XRAY_DIR, "test"))
    parser.add_argument("--output", default=os.path.join(RESULTS_DIR, "classifier_test_results.json"))
    args = parser.parse_args()

    device = config.DEVICE
    dataset = ChestXrayDataset(args.test_dir, transform=get_classification_test_transforms())
    assert dataset.class_names == config.CLASS_NAMES, \
        f"Class folders {dataset.class_names} do not match config {config.CLASS_NAMES}"
    print(dataset.summary())

    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    model = load_classifier(args.checkpoint, device)
    probs, targets = predict(model, loader, device)
    preds = probs.argmax(axis=1)

    metrics = compute_classification_metrics(preds, targets, probs, config.CLASS_NAMES)
    sens_spec = compute_sensitivity_specificity(preds, targets, positive_class=1)

    results = {
        "checkpoint": os.path.relpath(args.checkpoint, config.PROJECT_ROOT).replace("\\", "/"),
        "test_dir": os.path.relpath(args.test_dir, config.PROJECT_ROOT).replace("\\", "/"),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "n_samples": int(len(targets)),
        "class_counts": {name: int((targets == i).sum()) for i, name in enumerate(config.CLASS_NAMES)},
        "accuracy": metrics["accuracy"],
        "auc_roc": metrics.get("auc_roc"),
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        **sens_spec,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        save_path=os.path.join(config.VIZ_DIR, "test_confusion_matrix.png"),
        title=f"Test Set Confusion Matrix (n={len(targets)})",
    )
    plot_roc_curves(targets, probs, save_path=os.path.join(config.VIZ_DIR, "test_roc_curve.png"))

    print("\n" + metrics["classification_report"])
    for key in ["accuracy", "auc_roc", "sensitivity", "specificity", "ppv", "npv"]:
        print(f"  {key:<12} {results[key]:.4f}")
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()

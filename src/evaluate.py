"""
MedVision Model Evaluation
Comprehensive evaluation metrics for classification and segmentation models.
Includes AUC-ROC, confusion matrices, Dice scores, and error analysis visualization.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ──────────────────────────────────────────────
# Classification Metrics
# ──────────────────────────────────────────────

def compute_classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray = None,
    class_names: List[str] = None,
) -> Dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        predictions: Predicted class indices (N,)
        targets: Ground truth class indices (N,)
        probabilities: Prediction probabilities (N, num_classes) — for AUC-ROC
        class_names: Names for each class
    
    Returns:
        Dict with accuracy, precision, recall, f1, auc_roc, confusion_matrix
    """
    class_names = class_names or config.CLASS_NAMES
    
    metrics = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision_macro": float(precision_score(targets, predictions, average='macro', zero_division=0)),
        "recall_macro": float(recall_score(targets, predictions, average='macro', zero_division=0)),
        "f1_macro": float(f1_score(targets, predictions, average='macro', zero_division=0)),
        "confusion_matrix": confusion_matrix(targets, predictions).tolist(),
        "classification_report": classification_report(
            targets, predictions, target_names=class_names, zero_division=0
        ),
    }
    
    # Per-class metrics
    per_class = {}
    for i, name in enumerate(class_names):
        mask = targets == i
        if mask.sum() > 0:
            per_class[name] = {
                "precision": float(precision_score(targets == i, predictions == i, zero_division=0)),
                "recall": float(recall_score(targets == i, predictions == i, zero_division=0)),
                "f1": float(f1_score(targets == i, predictions == i, zero_division=0)),
                "support": int(mask.sum()),
            }
    metrics["per_class"] = per_class
    
    # AUC-ROC (requires probabilities)
    if probabilities is not None:
        try:
            if len(class_names) == 2:
                # Binary classification
                metrics["auc_roc"] = float(roc_auc_score(targets, probabilities[:, 1]))
            else:
                # Multi-class
                metrics["auc_roc"] = float(roc_auc_score(
                    targets, probabilities, multi_class='ovr', average='macro'
                ))
        except ValueError as e:
            metrics["auc_roc"] = None
            print(f"[WARN] Could not compute AUC-ROC: {e}")
    
    return metrics


def compute_sensitivity_specificity(
    predictions: np.ndarray,
    targets: np.ndarray,
    positive_class: int = 1,
) -> Dict[str, float]:
    """
    Compute sensitivity (TPR) and specificity (TNR).
    Critical clinical metrics for medical diagnostics.
    
    - Sensitivity = TP / (TP + FN) — ability to detect disease
    - Specificity = TN / (TN + FP) — ability to rule out disease
    """
    tp = np.sum((predictions == positive_class) & (targets == positive_class))
    tn = np.sum((predictions != positive_class) & (targets != positive_class))
    fp = np.sum((predictions == positive_class) & (targets != positive_class))
    fn = np.sum((predictions != positive_class) & (targets == positive_class))
    
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,  # Positive Predictive Value
        "npv": npv,  # Negative Predictive Value
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ──────────────────────────────────────────────
# Segmentation Metrics
# ──────────────────────────────────────────────

def compute_dice_score(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """
    Compute Dice coefficient between predicted and ground truth masks.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Standard metric for medical image segmentation quality.
    """
    pred_flat = pred_mask.flatten().astype(np.float32)
    true_flat = true_mask.flatten().astype(np.float32)
    
    intersection = np.sum(pred_flat * true_flat)
    union = np.sum(pred_flat) + np.sum(true_flat)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return float(dice)


def compute_iou(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """
    Compute Intersection over Union (IoU / Jaccard Index).
    
    IoU = |A ∩ B| / |A ∪ B|
    """
    pred_flat = pred_mask.flatten().astype(np.float32)
    true_flat = true_mask.flatten().astype(np.float32)
    
    intersection = np.sum(pred_flat * true_flat)
    union = np.sum(pred_flat) + np.sum(true_flat) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return float(iou)


def compute_pixel_accuracy(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
) -> float:
    """Compute pixel-level accuracy for segmentation."""
    correct = np.sum(pred_mask == true_mask)
    total = pred_mask.size
    return float(correct / total)


def compute_segmentation_metrics(
    pred_masks: List[np.ndarray],
    true_masks: List[np.ndarray],
) -> Dict:
    """
    Compute comprehensive segmentation metrics over a batch of masks.
    
    Returns:
        Dict with mean and std of Dice, IoU, and pixel accuracy
    """
    dice_scores = []
    iou_scores = []
    pixel_accuracies = []
    
    for pred, true in zip(pred_masks, true_masks):
        dice_scores.append(compute_dice_score(pred, true))
        iou_scores.append(compute_iou(pred, true))
        pixel_accuracies.append(compute_pixel_accuracy(pred, true))
    
    return {
        "dice_mean": float(np.mean(dice_scores)),
        "dice_std": float(np.std(dice_scores)),
        "iou_mean": float(np.mean(iou_scores)),
        "iou_std": float(np.std(iou_scores)),
        "pixel_accuracy_mean": float(np.mean(pixel_accuracies)),
        "pixel_accuracy_std": float(np.std(pixel_accuracies)),
        "num_samples": len(pred_masks),
    }


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
):
    """Plot a confusion matrix as a seaborn heatmap."""
    class_names = class_names or config.CLASS_NAMES
    save_path = save_path or os.path.join(config.VIZ_DIR, "confusion_matrix.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar_kws={'label': 'Count'},
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved: {save_path}")


def plot_roc_curves(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
):
    """
    Plot ROC curves for all classes.
    Shows model performance at various classification thresholds.
    """
    class_names = class_names or config.CLASS_NAMES
    save_path = save_path or os.path.join(config.VIZ_DIR, "roc_curves.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if len(class_names) == 2:
        # Binary ROC
        fpr, tpr, _ = roc_curve(targets, probabilities[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'{class_names[1]} (AUC = {roc_auc:.3f})')
    else:
        # Multi-class ROC (one-vs-rest)
        for i, name in enumerate(class_names):
            binary_targets = (targets == i).astype(int)
            if binary_targets.sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(binary_targets, probabilities[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — Chest X-Ray Classification', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curves saved: {save_path}")


def plot_training_curves(
    training_log: List[Dict],
    save_path: Optional[str] = None,
    model_name: str = "Model",
):
    """
    Plot training curves from training log.
    Shows loss, accuracy/AUC, and learning rate over epochs.
    """
    save_path = save_path or os.path.join(config.VIZ_DIR, "training_curves.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    epochs = [log['epoch'] for log in training_log]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss curve
    train_losses = [log.get('train_loss', 0) for log in training_log]
    val_losses = [log.get('val_loss', 0) for log in training_log]
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if any(v > 0 for v in val_losses):
        axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} — Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy/AUC curve
    val_accs = [log.get('val_accuracy', log.get('val_auc', 0)) for log in training_log]
    metric_name = 'AUC-ROC' if 'val_auc' in training_log[0] else 'Accuracy'
    axes[1].plot(epochs, val_accs, 'g-', label=f'Val {metric_name}', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel(metric_name)
    axes[1].set_title(f'{model_name} — {metric_name}', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate
    lrs = [log.get('lr_backbone', log.get('lr', 0)) for log in training_log]
    axes[2].plot(epochs, lrs, 'm-', label='Backbone LR', linewidth=2)
    if 'lr_head' in training_log[0]:
        head_lrs = [log.get('lr_head', 0) for log in training_log]
        axes[2].plot(epochs, head_lrs, 'c-', label='Head LR', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title(f'{model_name} — Learning Rate Schedule', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].ticklabel_format(style='scientific', axis='y', scilimits=(-4, -4))
    
    plt.suptitle(f'{model_name} Training Curves', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved: {save_path}")


def plot_segmentation_results(
    images: List[np.ndarray],
    true_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
    dice_scores: List[float] = None,
    save_path: Optional[str] = None,
    max_samples: int = 6,
):
    """
    Visualize segmentation results: image, ground truth, prediction, overlay.
    """
    save_path = save_path or os.path.join(config.VIZ_DIR, "segmentation_results.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    n = min(len(images), max_samples)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    
    if n == 1:
        axes = axes[np.newaxis, :]
    
    for i in range(n):
        # Original image
        img = images[i]
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        axes[i, 0].imshow(img, cmap='gray' if img.ndim == 2 else None)
        axes[i, 0].set_title("X-Ray", fontsize=10)
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(true_masks[i], cmap='gray')
        axes[i, 1].set_title("Ground Truth", fontsize=10)
        axes[i, 1].axis('off')
        
        # Predicted mask
        axes[i, 2].imshow(pred_masks[i], cmap='gray')
        dice_str = f" (Dice: {dice_scores[i]:.3f})" if dice_scores else ""
        axes[i, 2].set_title(f"Predicted{dice_str}", fontsize=10)
        axes[i, 2].axis('off')
        
        # Overlay
        if img.ndim == 2:
            overlay = np.stack([img, img, img], axis=-1)
        else:
            overlay = img.copy()
        if overlay.dtype != np.uint8:
            overlay = (overlay * 255).astype(np.uint8) if overlay.max() <= 1 else overlay.astype(np.uint8)
        
        # Color the prediction mask (green for lung regions)
        mask_colored = np.zeros_like(overlay)
        mask_colored[:, :, 1] = (pred_masks[i] * 128).astype(np.uint8)
        blended = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        axes[i, 3].imshow(blended)
        axes[i, 3].set_title("Overlay", fontsize=10)
        axes[i, 3].axis('off')
    
    plt.suptitle("Lung Segmentation Results", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Segmentation results saved: {save_path}")


# Import cv2 at module level for overlay
import cv2

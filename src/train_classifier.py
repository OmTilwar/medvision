"""
MedVision Classification Training Pipeline
Trains the chest X-ray classifier using transfer learning with early stopping.
"""

import os
import sys
import json
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.classifier import MedicalClassifier, build_classifier
from src.dataset import create_classification_dataloaders
from src.evaluate import compute_classification_metrics, compute_sensitivity_specificity


def train_one_epoch(
    model: MedicalClassifier,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = 0
    
    start_time = time.time()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False, ncols=100)
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{(preds == labels.cpu().numpy()).mean():.2%}',
        })
    
    elapsed = time.time() - start_time
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
    
    return {
        'loss': avg_loss,
        'accuracy': float(accuracy),
        'time': elapsed,
    }


@torch.no_grad()
def validate(
    model: MedicalClassifier,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Validate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    num_batches = 0
    
    for images, labels in tqdm(dataloader, desc="Validating", leave=False, ncols=100):
        images = images.to(device)
        labels = labels.to(device)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        num_batches += 1
        
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())
        all_probs.extend(probs)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # Compute metrics
    metrics = compute_classification_metrics(
        predictions=all_preds,
        targets=all_targets,
        probabilities=all_probs,
        class_names=config.CLASS_NAMES,
    )
    
    sens_spec = compute_sensitivity_specificity(all_preds, all_targets, positive_class=1)
    
    return {
        'loss': avg_loss,
        'accuracy': metrics['accuracy'],
        'auc_roc': metrics.get('auc_roc'),
        'precision': metrics['precision_macro'],
        'recall': metrics['recall_macro'],
        'f1': metrics['f1_macro'],
        'sensitivity': sens_spec['sensitivity'],
        'specificity': sens_spec['specificity'],
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs,
    }


def train(
    data_dir: Optional[str] = None,
    num_epochs: Optional[int] = None,
    resume_from: Optional[str] = None,
):
    """
    Full classification training pipeline.
    
    Args:
        data_dir: Path to chest X-ray data directory
        num_epochs: Override number of epochs
        resume_from: Path to checkpoint to resume from
    """
    num_epochs = num_epochs or config.NUM_EPOCHS_CLASSIFIER
    device = config.DEVICE
    
    print("=" * 60)
    print("  MedVision — Chest X-Ray Classification Training")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Backbone: {config.CLASSIFIER_BACKBONE}")
    print(f"  Classes: {config.NUM_CLASSES} ({', '.join(config.CLASS_NAMES)})")
    print("=" * 60)
    
    # ── Data ──
    train_loader, val_loader, test_loader, test_dataset = create_classification_dataloaders(
        data_dir=data_dir,
    )
    
    if len(train_loader.dataset) == 0:
        print("\n[ERROR] No training data found!")
        print(f"  Expected data at: {data_dir or config.CHEST_XRAY_DIR}")
        print("  Run: python scripts/download_data.py")
        return
    
    # ── Model ──
    if resume_from and os.path.exists(resume_from):
        from src.classifier import load_classifier
        model = load_classifier(resume_from, device)
        model.train()
    else:
        model = build_classifier(device)
    
    # ── Loss ──
    criterion = nn.CrossEntropyLoss()
    
    # ── Optimizer ──
    param_groups = model.get_parameter_groups()
    optimizer = torch.optim.Adam(param_groups, weight_decay=config.WEIGHT_DECAY)
    
    # ── Scheduler ──
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # ── Training Loop ──
    best_val_acc = 0.0
    patience_counter = 0
    training_log = []
    
    for epoch in range(num_epochs):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Log
        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_auc': val_metrics.get('auc_roc'),
            'val_sensitivity': val_metrics['sensitivity'],
            'val_specificity': val_metrics['specificity'],
            'lr_backbone': current_lrs[0],
            'lr_head': current_lrs[1] if len(current_lrs) > 1 else current_lrs[0],
            'train_time': train_metrics['time'],
        }
        training_log.append(epoch_log)
        
        auc_str = f"AUC: {val_metrics['auc_roc']:.4f} | " if val_metrics.get('auc_roc') else ""
        print(
            f"Epoch {epoch+1:3d}/{num_epochs} | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.2%} | "
            f"Val Acc: {val_metrics['accuracy']:.2%} | "
            f"{auc_str}"
            f"Sens: {val_metrics['sensitivity']:.2%} | "
            f"Spec: {val_metrics['specificity']:.2%} | "
            f"Time: {train_metrics['time']:.1f}s"
        )
        
        # ── Save best model ──
        val_score = val_metrics.get('auc_roc') or val_metrics['accuracy']
        if val_score > best_val_acc:
            best_val_acc = val_score
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_score': best_val_acc,
                'config': {
                    'num_classes': config.NUM_CLASSES,
                    'backbone': config.CLASSIFIER_BACKBONE,
                    'class_names': config.CLASS_NAMES,
                },
            }
            torch.save(checkpoint, config.CLASSIFIER_BEST_PATH)
            print(f"  ★ New best! Score: {best_val_acc:.4f} — saved")
        else:
            patience_counter += 1
        
        # ── Early stopping ──
        if patience_counter >= config.PATIENCE:
            print(f"\n[EARLY STOP] No improvement for {config.PATIENCE} epochs.")
            break
    
    # ── Save last model ──
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'best_val_score': best_val_acc,
    }, config.CLASSIFIER_LAST_PATH)
    
    # ── Save training log ──
    log_path = os.path.join(config.LOGS_DIR, "classifier_training_log.json")
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    # ── Final Test Evaluation ──
    print("\n" + "=" * 60)
    print("  Final Test Set Evaluation")
    print("=" * 60)
    
    # Load best model for test evaluation
    best_checkpoint = torch.load(config.CLASSIFIER_BEST_PATH, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, criterion, device)
    
    print(f"  Test Accuracy:   {test_metrics['accuracy']:.2%}")
    if test_metrics.get('auc_roc'):
        print(f"  Test AUC-ROC:    {test_metrics['auc_roc']:.4f}")
    print(f"  Sensitivity:     {test_metrics['sensitivity']:.2%}")
    print(f"  Specificity:     {test_metrics['specificity']:.2%}")
    print(f"  F1 Score:        {test_metrics['f1']:.4f}")
    print(f"\n  Best model: {config.CLASSIFIER_BEST_PATH}")
    print(f"  Training log: {log_path}")
    print("=" * 60)
    
    return model, training_log


if __name__ == "__main__":
    train()

"""
MedVision Segmentation Training Pipeline
Trains the U-Net for lung segmentation with Dice + BCE combined loss.
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
from src.segmenter import UNet, build_segmenter, BCEDiceLoss
from src.dataset import create_segmentation_dataloaders
from src.evaluate import compute_dice_score, compute_iou, compute_pixel_accuracy


def train_one_epoch(
    model: UNet,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    
    start_time = time.time()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False, ncols=100)
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward
        logits = model(images)
        loss = criterion(logits, masks)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Compute Dice for monitoring
        with torch.no_grad():
            pred_masks = (torch.sigmoid(logits) > 0.5).float()
            batch_dice = compute_dice_score(
                pred_masks.cpu().numpy(),
                masks.cpu().numpy(),
            )
        
        total_loss += loss.item()
        total_dice += batch_dice
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{batch_dice:.4f}',
        })
    
    elapsed = time.time() - start_time
    avg_loss = total_loss / max(num_batches, 1)
    avg_dice = total_dice / max(num_batches, 1)
    
    return {
        'loss': avg_loss,
        'dice': avg_dice,
        'time': elapsed,
    }


@torch.no_grad()
def validate(
    model: UNet,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Validate segmentation model."""
    model.eval()
    total_loss = 0.0
    all_dice = []
    all_iou = []
    all_pixel_acc = []
    num_batches = 0
    
    for images, masks in tqdm(dataloader, desc="Validating", leave=False, ncols=100):
        images = images.to(device)
        masks = masks.to(device)
        
        logits = model(images)
        loss = criterion(logits, masks)
        
        total_loss += loss.item()
        num_batches += 1
        
        # Threshold predictions
        pred_masks = (torch.sigmoid(logits) > 0.5).float()
        
        # Per-sample metrics
        for i in range(pred_masks.size(0)):
            pred_np = pred_masks[i].cpu().numpy().squeeze()
            true_np = masks[i].cpu().numpy().squeeze()
            
            all_dice.append(compute_dice_score(pred_np, true_np))
            all_iou.append(compute_iou(pred_np, true_np))
            all_pixel_acc.append(compute_pixel_accuracy(
                (pred_np > 0.5).astype(int),
                (true_np > 0.5).astype(int),
            ))
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'dice_mean': float(np.mean(all_dice)) if all_dice else 0.0,
        'dice_std': float(np.std(all_dice)) if all_dice else 0.0,
        'iou_mean': float(np.mean(all_iou)) if all_iou else 0.0,
        'pixel_accuracy': float(np.mean(all_pixel_acc)) if all_pixel_acc else 0.0,
    }


def train(
    data_dir: Optional[str] = None,
    num_epochs: Optional[int] = None,
):
    """
    Full segmentation training pipeline.
    """
    num_epochs = num_epochs or config.NUM_EPOCHS_SEGMENTER
    device = config.DEVICE
    
    print("=" * 60)
    print("  MedVision — Lung Segmentation Training (U-Net)")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Image size: {config.SEG_IMAGE_SIZE}x{config.SEG_IMAGE_SIZE}")
    print(f"  Loss: BCE + Dice (combined)")
    print("=" * 60)
    
    # ── Data ──
    train_loader, val_loader = create_segmentation_dataloaders(data_dir=data_dir)
    
    if len(train_loader.dataset) == 0:
        print("\n[ERROR] No segmentation data found!")
        print(f"  Expected data at: {data_dir or config.SEGMENTATION_DIR}")
        print("  Run: python scripts/download_data.py")
        return
    
    # ── Model ──
    model = build_segmenter(device)
    
    # ── Loss ──
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    
    # ── Optimizer ──
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LR_HEAD,
        weight_decay=config.WEIGHT_DECAY,
    )
    
    # ── Scheduler ──
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # ── Training Loop ──
    best_dice = 0.0
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
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_dice': train_metrics['dice'],
            'val_loss': val_metrics['loss'],
            'val_dice': val_metrics['dice_mean'],
            'val_iou': val_metrics['iou_mean'],
            'val_pixel_acc': val_metrics['pixel_accuracy'],
            'lr': optimizer.param_groups[0]['lr'],
            'train_time': train_metrics['time'],
        }
        training_log.append(epoch_log)
        
        print(
            f"Epoch {epoch+1:3d}/{num_epochs} | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"Train Dice: {train_metrics['dice']:.4f} | "
            f"Val Dice: {val_metrics['dice_mean']:.4f} ± {val_metrics['dice_std']:.4f} | "
            f"Val IoU: {val_metrics['iou_mean']:.4f} | "
            f"Time: {train_metrics['time']:.1f}s"
        )
        
        # ── Save best model ──
        if val_metrics['dice_mean'] > best_dice:
            best_dice = val_metrics['dice_mean']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'config': {
                    'in_channels': config.UNET_IN_CHANNELS,
                    'out_channels': config.UNET_OUT_CHANNELS,
                    'features': config.UNET_FEATURES,
                },
            }
            torch.save(checkpoint, config.SEGMENTER_BEST_PATH)
            print(f"  * New best Dice: {best_dice:.4f} - saved")
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
        'best_dice': best_dice,
    }, config.SEGMENTER_LAST_PATH)
    
    # ── Save training log ──
    log_path = os.path.join(config.LOGS_DIR, "segmenter_training_log.json")
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print("\n" + "=" * 60)
    print("  Segmentation Training Complete!")
    print("=" * 60)
    print(f"  Best Dice:       {best_dice:.4f}")
    print(f"  Best model:      {config.SEGMENTER_BEST_PATH}")
    print(f"  Training log:    {log_path}")
    print("=" * 60)
    
    return model, training_log


if __name__ == "__main__":
    train()

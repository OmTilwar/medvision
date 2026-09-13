"""
MedVision U-Net Segmentation Model
U-Net architecture for medical image segmentation (lung segmentation from chest X-rays).

Reference: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class DoubleConv(nn.Module):
    """
    Double convolution block: (Conv2d → BN → ReLU) × 2
    The fundamental building block of U-Net — two 3×3 convolutions
    with batch normalization and ReLU activation.
    """
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """
    Encoder block: MaxPool → DoubleConv
    Downsamples spatial dimensions by 2× and increases channel depth.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """
    Decoder block: Upsample → Concatenate (skip connection) → DoubleConv
    
    The skip connection from the encoder provides high-resolution features
    that help the decoder localize segmentation boundaries precisely.
    This is the key innovation of U-Net.
    """
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            # Bilinear upsampling (faster, less parameters)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Learnable transposed convolution
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input from previous decoder layer
            skip: Skip connection from corresponding encoder layer
        """
        x = self.up(x)
        
        # Handle size mismatch due to odd-sized inputs
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        
        # Concatenate skip connection
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1×1 convolution to map features to output classes."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for Medical Image Segmentation.
    
    Architecture:
        Input (1 × 256 × 256)  [grayscale X-ray]
            │
            ▼
        ┌─────────────────┐
        │  Encoder         │
        │  64 → 128 → 256 → 512  (DoubleConv + MaxPool at each level)
        └─────────────────┘
            │
            ▼
        ┌─────────────────┐
        │  Bottleneck      │
        │  512 → 1024      │
        └─────────────────┘
            │
            ▼
        ┌─────────────────┐
        │  Decoder         │  ← Skip connections from encoder
        │  512 → 256 → 128 → 64
        └─────────────────┘
            │
            ▼
        Output (1 × 256 × 256)  [binary lung mask]
    
    Key design:
        - Skip connections preserve fine-grained spatial info for precise boundaries
        - Bilinear upsampling (default) for efficiency, transposed conv as option
        - Binary output with sigmoid activation for lung segmentation
    """
    
    def __init__(
        self,
        in_channels: int = config.UNET_IN_CHANNELS,
        out_channels: int = config.UNET_OUT_CHANNELS,
        features: list = None,
        bilinear: bool = config.UNET_BILINEAR,
    ):
        super().__init__()
        
        features = features or config.UNET_FEATURES
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # ── Encoder ──
        self.inc = DoubleConv(in_channels, features[0])           # 1 → 64
        self.down1 = DownBlock(features[0], features[1])          # 64 → 128
        self.down2 = DownBlock(features[1], features[2])          # 128 → 256
        self.down3 = DownBlock(features[2], features[3])          # 256 → 512
        
        # ── Bottleneck ──
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(features[3], features[3] * 2 // factor)  # 512 → 1024 (or 512)
        
        # ── Decoder ──
        self.up1 = UpBlock(features[3] * 2 // factor + features[3], features[3] // factor, bilinear)
        self.up2 = UpBlock(features[3] // factor + features[2], features[2] // factor, bilinear)
        self.up3 = UpBlock(features[2] // factor + features[1], features[1] // factor, bilinear)
        self.up4 = UpBlock(features[1] // factor + features[0], features[0], bilinear)
        
        # ── Output ──
        self.outc = OutConv(features[0], out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: image → segmentation mask.
        
        Args:
            x: Input tensor (batch_size, in_channels, H, W)
            
        Returns:
            Logits tensor (batch_size, out_channels, H, W)
            Apply sigmoid externally for binary segmentation.
        """
        # Encoder (save intermediate outputs for skip connections)
        x1 = self.inc(x)       # (B, 64, H, W)
        x2 = self.down1(x1)    # (B, 128, H/2, W/2)
        x3 = self.down2(x2)    # (B, 256, H/4, W/4)
        x4 = self.down3(x3)    # (B, 512, H/8, W/8)
        x5 = self.down4(x4)    # (B, 1024, H/16, W/16)
        
        # Decoder (with skip connections)
        x = self.up1(x5, x4)   # (B, 512, H/8, W/8)
        x = self.up2(x, x3)    # (B, 256, H/4, W/4)
        x = self.up3(x, x2)    # (B, 128, H/2, W/2)
        x = self.up4(x, x1)    # (B, 64, H, W)
        
        logits = self.outc(x)   # (B, 1, H, W)
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with sigmoid activation for inference."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def count_parameters(self) -> dict:
        """Count parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total": total,
            "trainable": trainable,
            "total_mb": total * 4 / (1024 ** 2),
        }


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Measures overlap between predicted and ground truth masks.
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    DiceLoss = 1 - Dice
    
    Handles class imbalance better than BCE for segmentation tasks
    where background dominates foreground.
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # Flatten spatial dimensions
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice loss.
    
    BCE provides stable pixel-wise gradients.
    Dice optimizes the global overlap metric.
    Together they give better convergence than either alone.
    """
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice(pred, target)


def build_segmenter(device: torch.device = None) -> UNet:
    """Build and return the U-Net model."""
    device = device or config.DEVICE
    model = UNet(
        in_channels=config.UNET_IN_CHANNELS,
        out_channels=config.UNET_OUT_CHANNELS,
        features=config.UNET_FEATURES,
        bilinear=config.UNET_BILINEAR,
    )
    model = model.to(device)
    
    param_info = model.count_parameters()
    print(f"\nModel: U-Net Segmentation")
    print(f"  Input channels:   {config.UNET_IN_CHANNELS}")
    print(f"  Output channels:  {config.UNET_OUT_CHANNELS}")
    print(f"  Feature sizes:    {config.UNET_FEATURES}")
    print(f"  Total params:     {param_info['total']:,}")
    print(f"  Model size:       {param_info['total_mb']:.1f} MB (FP32)")
    print(f"  Device:           {device}\n")
    
    return model


def load_segmenter(checkpoint_path: str, device: torch.device = None) -> UNet:
    """Load a trained U-Net from checkpoint."""
    device = device or config.DEVICE
    model = UNet(
        in_channels=config.UNET_IN_CHANNELS,
        out_channels=config.UNET_OUT_CHANNELS,
        features=config.UNET_FEATURES,
    )
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Loaded U-Net from {checkpoint_path}")
    return model

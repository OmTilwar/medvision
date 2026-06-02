"""
Tests for MedVision U-Net Segmenter
Tests U-Net architecture, skip connections, loss functions, and mask output.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.segmenter import UNet, DoubleConv, DownBlock, UpBlock, DiceLoss, BCEDiceLoss, build_segmenter


class TestUNetComponents:
    """Test U-Net building blocks."""
    
    def test_double_conv(self):
        """DoubleConv should transform channels correctly."""
        block = DoubleConv(1, 64)
        x = torch.randn(1, 1, 64, 64)
        out = block(x)
        assert out.shape == (1, 64, 64, 64), f"Expected (1, 64, 64, 64), got {out.shape}"
    
    def test_down_block(self):
        """DownBlock should halve spatial dims and change channels."""
        block = DownBlock(64, 128)
        x = torch.randn(1, 64, 64, 64)
        out = block(x)
        assert out.shape == (1, 128, 32, 32), f"Expected (1, 128, 32, 32), got {out.shape}"


class TestUNet:
    """Test full U-Net model."""
    
    def test_forward_pass_shape(self):
        """U-Net output should match input spatial dimensions."""
        model = UNet(in_channels=1, out_channels=1)
        model.eval()
        
        x = torch.randn(2, 1, 256, 256)
        with torch.no_grad():
            out = model(x)
        
        assert out.shape == (2, 1, 256, 256), f"Expected (2, 1, 256, 256), got {out.shape}"
    
    def test_forward_pass_3channel(self):
        """U-Net should work with 3-channel input."""
        model = UNet(in_channels=3, out_channels=1)
        model.eval()
        
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = model(x)
        
        assert out.shape == (1, 1, 256, 256)
    
    def test_multi_class_output(self):
        """U-Net should support multi-class segmentation."""
        model = UNet(in_channels=1, out_channels=3)
        model.eval()
        
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            out = model(x)
        
        assert out.shape == (1, 3, 256, 256)
    
    def test_predict_sigmoid_range(self):
        """Predict should return values in [0, 1]."""
        model = UNet(in_channels=1, out_channels=1)
        model.eval()
        
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            mask = model.predict(x)
        
        assert mask.min() >= 0, "Sigmoid output should be >= 0"
        assert mask.max() <= 1, "Sigmoid output should be <= 1"
    
    def test_odd_input_size(self):
        """U-Net should handle non-power-of-2 input sizes via padding."""
        model = UNet(in_channels=1, out_channels=1)
        model.eval()
        
        # 240 is not a power of 2 but divisible by 16
        x = torch.randn(1, 1, 240, 240)
        with torch.no_grad():
            out = model(x)
        
        assert out.shape[2] == 240 and out.shape[3] == 240
    
    def test_parameter_count(self):
        """Model should have reasonable parameter count."""
        model = UNet(in_channels=1, out_channels=1)
        params = model.count_parameters()
        
        assert params["total"] > 0
        # U-Net with [64,128,256,512] should have ~7-8M params
        assert params["total"] > 1_000_000, "U-Net should have >1M parameters"
    
    def test_gradient_flow(self):
        """Gradients should flow through encoder, decoder, and skip connections."""
        model = UNet(in_channels=1, out_channels=1)
        model.train()
        
        x = torch.randn(1, 1, 256, 256)
        target = torch.rand(1, 1, 256, 256)
        
        output = model(x)
        loss = nn.BCEWithLogitsLoss()(output, target)
        loss.backward()
        
        # Check gradients exist in both encoder and decoder
        encoder_grad = any(p.grad is not None for p in model.inc.parameters())
        decoder_grad = any(p.grad is not None for p in model.up4.parameters())
        
        assert encoder_grad, "Encoder should have gradients"
        assert decoder_grad, "Decoder should have gradients"


class TestLossFunctions:
    """Test segmentation loss functions."""
    
    def test_dice_loss_perfect_match(self):
        """Dice loss should be ~0 for identical predictions and targets."""
        loss_fn = DiceLoss()
        pred = torch.ones(1, 1, 64, 64) * 100  # High logit → sigmoid ≈ 1
        target = torch.ones(1, 1, 64, 64)
        
        loss = loss_fn(pred, target)
        assert loss.item() < 0.01, f"Dice loss for perfect match should be near 0, got {loss.item()}"
    
    def test_dice_loss_no_overlap(self):
        """Dice loss should be ~1 for zero overlap."""
        loss_fn = DiceLoss()
        pred = torch.ones(1, 1, 64, 64) * -100  # Low logit → sigmoid ≈ 0
        target = torch.ones(1, 1, 64, 64)
        
        loss = loss_fn(pred, target)
        assert loss.item() > 0.9, f"Dice loss for no overlap should be near 1, got {loss.item()}"
    
    def test_bce_dice_combined(self):
        """BCEDiceLoss should combine both losses."""
        loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        pred = torch.randn(2, 1, 64, 64, requires_grad=True)
        target = torch.rand(2, 1, 64, 64).round()
        
        loss = loss_fn(pred, target)
        assert loss.item() > 0, "Combined loss should be positive"
        assert loss.requires_grad, "Loss should be differentiable"

"""
Tests for MedVision Chest X-Ray Classifier
Tests model architecture, forward pass, parameter counting, and checkpoint handling.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.classifier import MedicalClassifier, build_classifier


class TestMedicalClassifier:
    """Test suite for the MedicalClassifier model."""
    
    def test_model_creation(self):
        """Model should be created with default config."""
        model = MedicalClassifier(num_classes=2, pretrained=False)
        assert model is not None
        assert model.num_classes == 2
    
    def test_forward_pass_shape(self):
        """Forward pass should output correct shape (batch_size, num_classes)."""
        model = MedicalClassifier(num_classes=2, pretrained=False)
        model.eval()
        
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (4, 2), f"Expected (4, 2), got {output.shape}"
    
    def test_forward_pass_multi_class(self):
        """Forward pass should work with more than 2 classes."""
        model = MedicalClassifier(num_classes=14, pretrained=False)
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 14), f"Expected (2, 14), got {output.shape}"
    
    def test_predict_softmax(self):
        """Predict should return probabilities that sum to 1."""
        model = MedicalClassifier(num_classes=2, pretrained=False)
        model.eval()
        
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            probs = model.predict(x)
        
        assert probs.shape == (1, 2)
        assert abs(probs.sum().item() - 1.0) < 1e-5, "Probabilities should sum to 1"
        assert (probs >= 0).all(), "All probabilities should be non-negative"
    
    def test_get_features(self):
        """Feature extraction should return 512-dim vector for ResNet-18."""
        model = MedicalClassifier(num_classes=2, backbone_name="resnet18", pretrained=False)
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            features = model.get_features(x)
        
        assert features.shape == (2, 512), f"Expected (2, 512), got {features.shape}"
    
    def test_parameter_counting(self):
        """Parameter counting should return correct structure."""
        model = MedicalClassifier(num_classes=2, pretrained=False)
        params = model.count_parameters()
        
        assert "total" in params
        assert "trainable" in params
        assert "frozen" in params
        assert "total_mb" in params
        assert params["total"] > 0
        assert params["trainable"] > 0
        assert params["frozen"] >= 0
        assert params["total"] == params["trainable"] + params["frozen"]
    
    def test_layer_freezing(self):
        """Specified layers should have frozen parameters."""
        model = MedicalClassifier(num_classes=2, pretrained=False, freeze_layers=["layer1"])
        params = model.count_parameters()
        assert params["frozen"] > 0, "Some parameters should be frozen"
    
    def test_differential_lr_groups(self):
        """get_parameter_groups should return 2 groups with different LRs."""
        model = MedicalClassifier(num_classes=2, pretrained=False)
        groups = model.get_parameter_groups()
        
        assert len(groups) == 2, "Should have 2 parameter groups (backbone + head)"
        assert groups[0]["lr"] < groups[1]["lr"], "Backbone LR should be lower than head LR"
    
    def test_checkpoint_save_load(self):
        """Model should be saveable and loadable."""
        model = MedicalClassifier(num_classes=2, pretrained=False)
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': 1,
        }
        
        tmp_path = os.path.join(config.MODEL_DIR, "test_classifier.pth")
        torch.save(checkpoint, tmp_path)
        
        # Load checkpoint
        model2 = MedicalClassifier(num_classes=2, pretrained=False)
        loaded = torch.load(tmp_path, map_location='cpu', weights_only=True)
        model2.load_state_dict(loaded['model_state_dict'])
        
        # Verify outputs match
        x = torch.randn(1, 3, 224, 224)
        model.eval()
        model2.eval()
        
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        
        assert torch.allclose(out1, out2, atol=1e-6), "Loaded model should produce same outputs"
        
        # Cleanup
        os.remove(tmp_path)
    
    def test_gradient_flow(self):
        """Gradients should flow through the model."""
        model = MedicalClassifier(num_classes=2, pretrained=False)
        model.train()
        
        x = torch.randn(2, 3, 224, 224)
        labels = torch.tensor([0, 1])
        
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, labels)
        loss.backward()
        
        # Check that head parameters have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.classifier_head.parameters()
        )
        assert has_grad, "Head parameters should have gradients"

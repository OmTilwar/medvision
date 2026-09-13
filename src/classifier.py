"""
MedVision Chest X-Ray Classifier
ResNet-18 based multi-class medical image classifier.
Fine-tuned from ImageNet weights for chest X-ray disease detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class MedicalClassifier(nn.Module):
    """
    Medical Image Classifier using transfer learning.
    
    Architecture:
        Input (3 × 224 × 224)
            → ResNet-18 backbone (ImageNet pre-trained, layer1 frozen)
            → 512-dim features (AdaptiveAvgPool)
            → FC Head: Linear(512→256) → BN → ReLU → Dropout(0.3) → Linear(256→num_classes)
            → Sigmoid for multi-label or Softmax for multi-class
    
    Design decisions:
        - Grayscale→3ch conversion in preprocessing (not here) for clean separation
        - Higher dropout (0.3) than AuthNet (0.2) since medical datasets are smaller
        - Layer1 frozen to prevent overfitting on limited medical data
    """
    
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        backbone_name: str = config.CLASSIFIER_BACKBONE,
        pretrained: bool = config.CLASSIFIER_PRETRAINED,
        freeze_layers: list = None,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        freeze_layers = freeze_layers or config.FREEZE_LAYERS
        
        # ── Backbone ──
        if backbone_name == "resnet18":
            backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            backbone_dim = 512
        elif backbone_name == "resnet34":
            backbone = models.resnet34(
                weights=models.ResNet34_Weights.DEFAULT if pretrained else None
            )
            backbone_dim = 512
        elif backbone_name == "resnet50":
            backbone = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Extract feature layers (remove final FC)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Store layer4 reference for Grad-CAM
        self.layer4 = backbone.layer4
        
        # ── Classification Head ──
        self.classifier_head = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )
        
        # ── Freeze early layers ──
        self._freeze_layers(freeze_layers)
        
        self.backbone_dim = backbone_dim
    
    def _freeze_layers(self, layer_names: list):
        """Freeze specified backbone layers to prevent overfitting."""
        for name, param in self.backbone.named_parameters():
            for freeze_name in layer_names:
                layer_map = {
                    "conv1": "0.", "bn1": "1.",
                    "layer1": "4.", "layer2": "5.",
                }
                if freeze_name in layer_map and name.startswith(layer_map[freeze_name]):
                    param.requires_grad = False
                    break
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features before classification head."""
        features = self.backbone(x)
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: image → class logits.
        
        Args:
            x: Input tensor (batch_size, 3, 224, 224)
            
        Returns:
            Logits tensor (batch_size, num_classes) — apply sigmoid/softmax externally
        """
        features = self.get_features(x)
        logits = self.classifier_head(features)
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with softmax for inference."""
        logits = self.forward(x)
        if self.num_classes == 2:
            return F.softmax(logits, dim=1)
        return torch.sigmoid(logits)
    
    def get_parameter_groups(self) -> list:
        """Differential learning rates: low for backbone, high for head."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.classifier_head.parameters())
        
        return [
            {"params": backbone_params, "lr": config.LR_BACKBONE},
            {"params": head_params, "lr": config.LR_HEAD},
        ]
    
    def count_parameters(self) -> dict:
        """Count total, trainable, and frozen parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "total_mb": total * 4 / (1024 ** 2),
        }


def build_classifier(device: torch.device = None) -> MedicalClassifier:
    """Build and return the MedicalClassifier on the specified device."""
    device = device or config.DEVICE
    model = MedicalClassifier(
        num_classes=config.NUM_CLASSES,
        backbone_name=config.CLASSIFIER_BACKBONE,
        pretrained=config.CLASSIFIER_PRETRAINED,
    )
    model = model.to(device)
    
    param_info = model.count_parameters()
    print(f"\nModel: MedicalClassifier ({config.CLASSIFIER_BACKBONE})")
    print(f"  Classes:          {config.NUM_CLASSES} ({', '.join(config.CLASS_NAMES)})")
    print(f"  Total params:     {param_info['total']:,}")
    print(f"  Trainable params: {param_info['trainable']:,}")
    print(f"  Frozen params:    {param_info['frozen']:,}")
    print(f"  Model size:       {param_info['total_mb']:.1f} MB (FP32)")
    print(f"  Device:           {device}\n")
    
    return model


def load_classifier(checkpoint_path: str, device: torch.device = None) -> MedicalClassifier:
    """Load a trained classifier from checkpoint."""
    device = device or config.DEVICE
    model = MedicalClassifier(
        num_classes=config.NUM_CLASSES,
        backbone_name=config.CLASSIFIER_BACKBONE,
        pretrained=False,
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
    
    print(f"Loaded classifier from {checkpoint_path}")
    return model

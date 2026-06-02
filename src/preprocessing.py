"""
MedVision Medical Image Preprocessing
Specialized preprocessing transforms for clinical imaging workflows.
Includes CLAHE, CT windowing, medical-grade augmentation, and normalization.
"""

import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class CLAHETransform:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Standard preprocessing for medical X-rays to enhance local contrast
    while preventing noise amplification — widely used in radiology AI pipelines.
    
    Reference: Pizer et al., "Adaptive Histogram Equalization and Its Variations"
    """
    
    def __init__(
        self,
        clip_limit: float = config.CLAHE_CLIP_LIMIT,
        tile_grid_size: int = config.CLAHE_TILE_SIZE,
    ):
        self.clip_limit = clip_limit
        self.tile_grid_size = (tile_grid_size, tile_grid_size)
    
    def __call__(self, image: Image.Image) -> Image.Image:
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            # Convert to LAB color space, apply CLAHE to L channel
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_grid_size,
            )
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale — apply CLAHE directly
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_grid_size,
            )
            result = clahe.apply(img_array)
        
        return Image.fromarray(result)


class CTWindowing:
    """
    CT Windowing Transform.
    Maps raw Hounsfield Unit (HU) values to a displayable range based on
    clinical presets (lung, bone, soft tissue, brain).
    
    This is critical for CT imaging — different tissue types require 
    different windowing to be visible.
    """
    
    def __init__(self, window_center: int, window_width: int):
        self.window_center = window_center
        self.window_width = window_width
    
    def __call__(self, pixel_array: np.ndarray) -> np.ndarray:
        """Apply windowing to HU values."""
        min_val = self.window_center - self.window_width // 2
        max_val = self.window_center + self.window_width // 2
        
        windowed = np.clip(pixel_array, min_val, max_val)
        windowed = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        return windowed
    
    @classmethod
    def lung(cls) -> "CTWindowing":
        """Lung window preset — optimized for lung parenchyma visualization."""
        return cls(*config.LUNG_WINDOW)
    
    @classmethod
    def bone(cls) -> "CTWindowing":
        """Bone window preset — optimized for skeletal structures."""
        return cls(*config.BONE_WINDOW)
    
    @classmethod
    def soft_tissue(cls) -> "CTWindowing":
        """Soft tissue window preset — optimized for organs and soft tissue."""
        return cls(*config.SOFT_TISSUE_WINDOW)
    
    @classmethod
    def brain(cls) -> "CTWindowing":
        """Brain window preset — optimized for intracranial structures."""
        return cls(*config.BRAIN_WINDOW)


class GrayscaleTo3Channel:
    """
    Convert grayscale images to 3-channel for pretrained backbone compatibility.
    Medical X-rays are typically grayscale, but ImageNet-pretrained models 
    expect 3-channel input. We replicate the grayscale channel 3x.
    """
    
    def __call__(self, image: Image.Image) -> Image.Image:
        if image.mode == "L":
            return image.convert("RGB")
        return image


class MedicalNormalize:
    """
    Medical image normalization.
    Offers both ImageNet normalization (for transfer learning) and 
    per-dataset normalization (for training from scratch).
    """
    
    # ImageNet statistics (used when fine-tuning pretrained backbones)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Chest X-ray approximate statistics (from CheXpert/MIMIC-CXR)
    CHEST_XRAY_MEAN = [0.5056, 0.5056, 0.5056]
    CHEST_XRAY_STD = [0.252, 0.252, 0.252]
    
    @classmethod
    def imagenet(cls) -> transforms.Normalize:
        return transforms.Normalize(mean=cls.IMAGENET_MEAN, std=cls.IMAGENET_STD)
    
    @classmethod
    def chest_xray(cls) -> transforms.Normalize:
        return transforms.Normalize(mean=cls.CHEST_XRAY_MEAN, std=cls.CHEST_XRAY_STD)


def get_classification_train_transforms() -> transforms.Compose:
    """
    Training transforms for chest X-ray classification.
    Includes medical-specific augmentations that preserve clinical features:
    - Mild rotation (±10°) — X-rays can be slightly rotated
    - Horizontal flip — valid for chest X-rays (left-right symmetry)
    - NO vertical flip — anatomically invalid for chest X-rays
    - CLAHE — enhances contrast in underexposed regions
    - Mild brightness/contrast jitter — simulates exposure variation
    """
    return transforms.Compose([
        GrayscaleTo3Channel(),
        CLAHETransform(clip_limit=2.0, tile_grid_size=8),
        transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
        transforms.RandomCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        MedicalNormalize.imagenet(),
    ])


def get_classification_test_transforms() -> transforms.Compose:
    """Test/inference transforms — deterministic, no augmentation."""
    return transforms.Compose([
        GrayscaleTo3Channel(),
        CLAHETransform(clip_limit=2.0, tile_grid_size=8),
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        MedicalNormalize.imagenet(),
    ])


def get_segmentation_train_transforms(
    image_size: int = None,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Paired transforms for segmentation training.
    Returns separate transform pipelines for image and mask that must be
    applied with the same random seed to maintain spatial correspondence.
    
    Returns:
        (image_transform, mask_transform)
    """
    image_size = image_size or config.SEG_IMAGE_SIZE
    
    image_transform = transforms.Compose([
        GrayscaleTo3Channel(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        MedicalNormalize.imagenet(),
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    
    return image_transform, mask_transform


def get_segmentation_test_transforms(
    image_size: int = None,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Paired test transforms for segmentation (no augmentation)."""
    image_size = image_size or config.SEG_IMAGE_SIZE
    
    image_transform = transforms.Compose([
        GrayscaleTo3Channel(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        MedicalNormalize.imagenet(),
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    
    return image_transform, mask_transform


def paired_random_transform(
    image: Image.Image,
    mask: Image.Image,
    image_size: int = None,
    is_train: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply synchronized random augmentations to image-mask pairs.
    
    Critical for segmentation: the same spatial transforms (flip, rotate, crop) 
    must be applied to BOTH the image and its corresponding mask.
    """
    image_size = image_size or config.SEG_IMAGE_SIZE
    
    # Convert grayscale to RGB for pretrained backbone
    if image.mode == "L":
        image = image.convert("RGB")
    if mask.mode != "L":
        mask = mask.convert("L")
    
    # Resize both
    image = image.resize((image_size, image_size), Image.BILINEAR)
    mask = mask.resize((image_size, image_size), Image.NEAREST)
    
    if is_train:
        # Synchronized random horizontal flip
        if np.random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Synchronized random rotation (small angle for medical images)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-10, 10)
            image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
            mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)
    
    # Convert to tensors
    image_tensor = transforms.ToTensor()(image)
    mask_tensor = transforms.ToTensor()(mask)
    
    # Normalize image (not mask!)
    image_tensor = transforms.Normalize(
        mean=MedicalNormalize.IMAGENET_MEAN,
        std=MedicalNormalize.IMAGENET_STD,
    )(image_tensor)
    
    # Binarize mask (threshold at 0.5)
    mask_tensor = (mask_tensor > 0.5).float()
    
    return image_tensor, mask_tensor


def inverse_normalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized tensor back to a displayable numpy image (H, W, C)."""
    mean = torch.tensor(MedicalNormalize.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(MedicalNormalize.IMAGENET_STD).view(3, 1, 1)
    
    tensor = tensor.cpu().clone()
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)
    
    image = tensor.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    return image

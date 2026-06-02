"""
MedVision Dataset Pipeline
Dataset classes for chest X-ray classification and lung segmentation.
Supports folder-based image loading and DICOM ingestion.
"""

import os
import random
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.preprocessing import (
    get_classification_train_transforms,
    get_classification_test_transforms,
    paired_random_transform,
)


class ChestXrayDataset(Dataset):
    """
    Chest X-ray Classification Dataset.
    
    Loads images from class-based subdirectories (e.g., NORMAL/, PNEUMONIA/).
    Handles grayscale medical images and converts to 3-channel for 
    pretrained backbone compatibility.
    
    Expected structure:
        root_dir/
            NORMAL/
                img1.jpeg
                img2.jpeg
            PNEUMONIA/
                img3.jpeg
                ...
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        min_samples_per_class: int = 2,
    ):
        self.root_dir = root_dir
        self.transform = transform or get_classification_test_transforms()
        
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        self.class_names: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        
        if not os.path.exists(root_dir):
            print(f"[WARN] Dataset directory not found: {root_dir}")
            return
        
        # Scan class directories
        class_dirs = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        class_idx = 0
        
        for class_name in class_dirs:
            class_path = os.path.join(root_dir, class_name)
            images = [
                f for f in os.listdir(class_path)
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]
            
            if len(images) < min_samples_per_class:
                continue
            
            self.class_names.append(class_name)
            self.class_to_idx[class_name] = class_idx
            
            for img_name in sorted(images):
                self.image_paths.append(os.path.join(class_path, img_name))
                self.labels.append(class_idx)
            
            class_idx += 1
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to load {img_path}: {e}")
            # Fallback: random valid image from same class
            same_class = [i for i, l in enumerate(self.labels) if l == label and i != idx]
            if same_class:
                return self.__getitem__(random.choice(same_class))
            image = Image.new("RGB", (config.IMAGE_SIZE, config.IMAGE_SIZE))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_labels(self) -> List[int]:
        """Return all labels for sampler creation."""
        return self.labels
    
    def summary(self) -> str:
        """Print dataset summary."""
        n_classes = len(self.class_names)
        n_images = len(self.image_paths)
        if n_images == 0:
            return f"Empty dataset at {self.root_dir}"
        
        class_counts = {}
        for label in self.labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        counts = list(class_counts.values())
        class_details = ", ".join(
            f"{self.class_names[i]}: {class_counts.get(i, 0)}"
            for i in range(n_classes)
        )
        return (
            f"Dataset: {self.root_dir}\n"
            f"  Classes: {n_classes} ({class_details})\n"
            f"  Images:  {n_images}\n"
            f"  Samples/class: min={min(counts)}, max={max(counts)}, "
            f"mean={sum(counts)/len(counts):.1f}"
        )


class SegmentationDataset(Dataset):
    """
    Medical Image Segmentation Dataset.
    
    Loads paired image-mask data for lung segmentation.
    Applies synchronized random transforms to maintain spatial correspondence.
    
    Expected structure:
        root_dir/
            images/
                img001.png
                img002.png
            masks/
                img001.png  (binary mask, same filename)
                img002.png
    """
    
    def __init__(
        self,
        root_dir: str,
        is_train: bool = True,
        image_size: int = None,
    ):
        self.root_dir = root_dir
        self.is_train = is_train
        self.image_size = image_size or config.SEG_IMAGE_SIZE
        
        self.image_paths: List[str] = []
        self.mask_paths: List[str] = []
        
        images_dir = os.path.join(root_dir, "images")
        masks_dir = os.path.join(root_dir, "masks")
        
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            print(f"[WARN] Segmentation data not found at {root_dir}")
            return
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Match images to masks by filename
        for img_name in sorted(os.listdir(images_dir)):
            if os.path.splitext(img_name)[1].lower() not in valid_extensions:
                continue
            
            img_path = os.path.join(images_dir, img_name)
            
            # Look for matching mask with same stem
            stem = os.path.splitext(img_name)[0]
            mask_path = None
            for ext in valid_extensions:
                candidate = os.path.join(masks_dir, stem + ext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break
            
            if mask_path:
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            image = Image.open(self.image_paths[idx])
            mask = Image.open(self.mask_paths[idx])
        except Exception as e:
            print(f"[WARN] Failed to load pair {idx}: {e}")
            # Return blank pair
            image = Image.new("L", (self.image_size, self.image_size))
            mask = Image.new("L", (self.image_size, self.image_size))
        
        # Apply synchronized transforms
        image_tensor, mask_tensor = paired_random_transform(
            image, mask,
            image_size=self.image_size,
            is_train=self.is_train,
        )
        
        return image_tensor, mask_tensor
    
    def summary(self) -> str:
        return (
            f"Segmentation Dataset: {self.root_dir}\n"
            f"  Image-Mask pairs: {len(self)}\n"
            f"  Image size: {self.image_size}x{self.image_size}\n"
            f"  Mode: {'train' if self.is_train else 'test'}"
        )


def create_classification_dataloaders(
    data_dir: str = None,
    batch_size: int = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, ChestXrayDataset]:
    """
    Create train/val/test DataLoaders for chest X-ray classification.
    
    Expects data_dir to have train/, val/, test/ subdirectories,
    each containing class folders (NORMAL/, PNEUMONIA/).
    
    Returns:
        (train_loader, val_loader, test_loader, test_dataset)
    """
    data_dir = data_dir or config.CHEST_XRAY_DIR
    batch_size = batch_size or config.BATCH_SIZE
    
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    
    # Create datasets
    train_dataset = ChestXrayDataset(
        root_dir=train_dir,
        transform=get_classification_train_transforms(),
    )
    
    val_dataset = ChestXrayDataset(
        root_dir=val_dir,
        transform=get_classification_test_transforms(),
    )
    
    test_dataset = ChestXrayDataset(
        root_dir=test_dir,
        transform=get_classification_test_transforms(),
    )
    
    print(f"\n{'='*60}")
    print("Classification Dataset Summary")
    print(f"{'='*60}")
    print(f"[Train] {train_dataset.summary()}")
    print(f"[Val]   {val_dataset.summary()}")
    print(f"[Test]  {test_dataset.summary()}")
    print(f"{'='*60}\n")
    
    # Handle class imbalance with weighted sampling
    if len(train_dataset) > 0:
        class_counts = {}
        for label in train_dataset.labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        weights = [1.0 / class_counts[label] for label in train_dataset.labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader, test_dataset


def create_segmentation_dataloaders(
    data_dir: str = None,
    batch_size: int = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val DataLoaders for segmentation.
    
    Splits the dataset 80/20 if no explicit split directories exist.
    
    Returns:
        (train_loader, val_loader)
    """
    data_dir = data_dir or config.SEGMENTATION_DIR
    batch_size = batch_size or config.BATCH_SIZE
    
    # Check for pre-split directories
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        train_dataset = SegmentationDataset(root_dir=train_dir, is_train=True)
        val_dataset = SegmentationDataset(root_dir=val_dir, is_train=False)
    else:
        # Single directory — split automatically
        full_dataset = SegmentationDataset(root_dir=data_dir, is_train=True)
        n_total = len(full_dataset)
        n_train = int(n_total * 0.8)
        n_val = n_total - n_train
        
        train_dataset, val_dataset = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
    
    print(f"\n{'='*60}")
    print("Segmentation Dataset Summary")
    print(f"{'='*60}")
    if hasattr(train_dataset, 'summary'):
        print(f"[Train] {train_dataset.summary()}")
        print(f"[Val]   {val_dataset.summary()}")
    else:
        print(f"[Train] {len(train_dataset)} pairs")
        print(f"[Val]   {len(val_dataset)} pairs")
    print(f"{'='*60}\n")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    
    return train_loader, val_loader

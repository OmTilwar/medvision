"""
MedVision Model Interpretability
Grad-CAM visualization for medical image classification decisions.
Adapted from AuthNet's interpretability module for clinical imaging.

Shows which regions of a chest X-ray the model focuses on when
predicting specific diseases — critical for clinical trust and validation.
"""

import os
import sys
from typing import Optional, List, Tuple

import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.classifier import MedicalClassifier
from src.preprocessing import (
    get_classification_test_transforms,
    inverse_normalize,
)

# Conditional import
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenGradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("[WARN] pytorch-grad-cam not installed. Interpretability features disabled.")


def get_gradcam(
    model: MedicalClassifier,
    method: str = "gradcam",
) -> "GradCAM":
    """
    Create a Grad-CAM instance for the classifier.
    
    Targets the last convolutional layer of ResNet-18's layer4,
    where the highest-level features are computed.
    """
    if not GRADCAM_AVAILABLE:
        raise RuntimeError("pytorch-grad-cam is required. Install: pip install pytorch-grad-cam")
    
    # Target the last conv block of the backbone
    target_layers = [model.backbone[7][-1]]  # layer4[-1]
    
    cam_class = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "eigengradcam": EigenGradCAM,
    }.get(method, GradCAM)
    
    return cam_class(model=model, target_layers=target_layers)


def generate_heatmap(
    model: MedicalClassifier,
    image_input,
    target_class: Optional[int] = None,
    method: str = "gradcam",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a Grad-CAM heatmap for a medical image.
    
    Shows which regions the model focuses on when predicting a specific disease.
    
    Args:
        model: Trained MedicalClassifier
        image_input: Image path, PIL Image, or tensor
        target_class: Class index to explain (None = predicted class)
        method: CAM method ("gradcam", "gradcam++", "eigengradcam")
    
    Returns:
        (original_image, heatmap_colored, overlay) as numpy arrays (HWC, uint8)
    """
    if not GRADCAM_AVAILABLE:
        raise RuntimeError("pytorch-grad-cam is required.")
    
    model.eval()
    device = next(model.parameters()).device
    transform = get_classification_test_transforms()
    
    # Load and preprocess image
    if isinstance(image_input, str):
        original_pil = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        original_pil = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        original_pil = image_input.convert("RGB")
    elif isinstance(image_input, torch.Tensor):
        input_tensor = image_input.unsqueeze(0) if image_input.dim() == 3 else image_input
        original_np = inverse_normalize(image_input if image_input.dim() == 3 else image_input[0])
        original_pil = Image.fromarray(original_np)
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")
    
    # Resize for display
    original_resized = original_pil.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    original_np = np.array(original_resized).astype(np.float32) / 255.0
    
    # Transform for model
    if not isinstance(image_input, torch.Tensor):
        input_tensor = transform(original_pil).unsqueeze(0).to(device)
    else:
        input_tensor = input_tensor.to(device)
    
    # Get prediction if target class not specified
    if target_class is None:
        with torch.no_grad():
            logits = model(input_tensor)
            target_class = logits.argmax(dim=1).item()
    
    # Create Grad-CAM
    cam = get_gradcam(model, method)
    targets = [ClassifierOutputTarget(target_class)]
    
    # Generate heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]
    
    # Create overlay
    overlay = show_cam_on_image(original_np, grayscale_cam, use_rgb=True)
    
    # Colored heatmap
    heatmap_colored = cv2.applyColorMap(
        (grayscale_cam * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    original_uint8 = (original_np * 255).astype(np.uint8)
    
    return original_uint8, heatmap_colored, overlay


def visualize_prediction(
    model: MedicalClassifier,
    image_input,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    method: str = "gradcam",
):
    """
    Visualize a classification prediction with Grad-CAM explanation.
    
    Creates a figure with:
    - Original image
    - Grad-CAM heatmap for predicted class
    - Prediction confidence bar chart
    
    This is the "explainable AI" visualization clinicians need to trust model outputs.
    """
    class_names = class_names or config.CLASS_NAMES
    save_path = save_path or os.path.join(config.VIZ_DIR, "prediction_explanation.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    transform = get_classification_test_transforms()
    
    # Load image
    if isinstance(image_input, str):
        pil_image = Image.open(image_input).convert("RGB")
    else:
        pil_image = image_input.convert("RGB") if isinstance(image_input, Image.Image) else Image.fromarray(image_input)
    
    # Get prediction
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    
    predicted_class = probs.argmax()
    predicted_name = class_names[predicted_class] if predicted_class < len(class_names) else f"Class {predicted_class}"
    
    # Generate Grad-CAM for predicted class
    original, heatmap, overlay = generate_heatmap(
        model, pil_image, target_class=predicted_class, method=method
    )
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    fig.suptitle(
        f"Prediction: {predicted_name} ({probs[predicted_class]:.1%} confidence)",
        fontsize=14, fontweight='bold',
        color='green' if probs[predicted_class] > 0.7 else 'orange'
    )
    
    # Original
    axes[0].imshow(original)
    axes[0].set_title("Chest X-Ray", fontsize=11)
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap)
    axes[1].set_title(f"Grad-CAM: {predicted_name}", fontsize=11)
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Attention Overlay", fontsize=11)
    axes[2].axis('off')
    
    # Confidence bars
    colors = ['#2ecc71' if i == predicted_class else '#95a5a6' for i in range(len(probs))]
    bars = axes[3].barh(
        [class_names[i] if i < len(class_names) else f"Class {i}" for i in range(len(probs))],
        probs * 100,
        color=colors,
    )
    axes[3].set_xlabel("Confidence (%)")
    axes[3].set_title("Prediction Confidence", fontsize=11)
    axes[3].set_xlim(0, 100)
    
    # Add percentage labels
    for bar, prob in zip(bars, probs):
        axes[3].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f'{prob:.1%}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Prediction visualization saved: {save_path}")
    print(f"  Predicted: {predicted_name} ({probs[predicted_class]:.1%})")


def generate_class_activation_comparison(
    model: MedicalClassifier,
    image_input,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    method: str = "gradcam",
):
    """
    Generate Grad-CAM heatmaps for ALL classes on a single image.
    
    Shows how the model's attention shifts depending on which disease
    it's "looking for" — a powerful interpretability tool for clinicians.
    """
    class_names = class_names or config.CLASS_NAMES
    save_path = save_path or os.path.join(config.VIZ_DIR, "class_comparison.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    num_classes = len(class_names)
    fig, axes = plt.subplots(1, num_classes + 1, figsize=((num_classes + 1) * 4, 4))
    
    # Original image
    if isinstance(image_input, str):
        pil_image = Image.open(image_input).convert("RGB")
    else:
        pil_image = image_input
    
    original = np.array(pil_image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))
    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis('off')
    
    # Grad-CAM for each class
    for i, class_name in enumerate(class_names):
        try:
            _, _, overlay = generate_heatmap(model, pil_image, target_class=i, method=method)
            axes[i + 1].imshow(overlay)
            axes[i + 1].set_title(f"Grad-CAM: {class_name}", fontsize=10)
        except Exception as e:
            axes[i + 1].text(0.5, 0.5, f"Error: {str(e)[:30]}", ha='center', va='center')
            axes[i + 1].set_title(f"{class_name} (error)", fontsize=10)
        axes[i + 1].axis('off')
    
    plt.suptitle("Per-Class Activation Maps", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Class activation comparison saved: {save_path}")


def generate_batch_heatmaps(
    model: MedicalClassifier,
    image_paths: List[str],
    class_names: List[str] = None,
    save_dir: Optional[str] = None,
    method: str = "gradcam",
    max_images: int = 12,
):
    """
    Generate Grad-CAM heatmaps for a batch of images in a grid.
    Shows original + overlay side-by-side for visual inspection.
    """
    class_names = class_names or config.CLASS_NAMES
    save_dir = save_dir or config.VIZ_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    image_paths = image_paths[:max_images]
    n = len(image_paths)
    
    if n == 0:
        print("No images to process.")
        return
    
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 6, rows * 3))
    if rows == 1:
        axes = axes[np.newaxis, :]
    
    model.eval()
    device = next(model.parameters()).device
    transform = get_classification_test_transforms()
    
    for i, img_path in enumerate(image_paths):
        row = i // cols
        col = i % cols
        
        try:
            # Get prediction
            pil_image = Image.open(img_path).convert("RGB")
            input_tensor = transform(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(input_tensor)
                predicted = logits.argmax(dim=1).item()
                prob = torch.softmax(logits, dim=1)[0, predicted].item()
            
            pred_name = class_names[predicted] if predicted < len(class_names) else f"Class {predicted}"
            
            original, _, overlay = generate_heatmap(model, img_path, method=method)
            
            axes[row, col * 2].imshow(original)
            axes[row, col * 2].set_title(f"{pred_name} ({prob:.0%})", fontsize=8)
            axes[row, col * 2].axis('off')
            
            axes[row, col * 2 + 1].imshow(overlay)
            axes[row, col * 2 + 1].set_title("Grad-CAM", fontsize=8)
            axes[row, col * 2 + 1].axis('off')
        except Exception as e:
            print(f"  [WARN] Failed for {img_path}: {e}")
    
    # Hide unused subplots
    for i in range(n, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col * 2].axis('off')
        axes[row, col * 2 + 1].axis('off')
    
    plt.suptitle("MedVision Grad-CAM: Disease Detection Attention Maps",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    grid_path = os.path.join(save_dir, "gradcam_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Grad-CAM grid saved: {grid_path}")

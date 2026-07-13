# src/dataset.py
import os
from typing import List, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from src import config

class DentalDataset(Dataset):
    """
    optimized, aspect-ratio preserving PyTorch Dataset for clinical dental photographs.
    applies zero-padded letterboxing and outputs float images and one-hot target masks.
    """
    def __init__(self, image_paths: List[str], mask_paths: List[str], is_training: bool = True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        
        # anatomically-preserving letterboxing pipeline
        self.transform = A.Compose([
            # resize longest side to target limit (preserves aspect ratio)
            A.LongestMaxSize(max_size=config.TARGET_SIZE[0]),
            # pad shorter side with zero-pixels to hit target shape
            A.PadIfNeeded(
                min_height=config.TARGET_SIZE[0], 
                min_width=config.TARGET_SIZE[1], 
                border_mode=cv2.BORDER_CONSTANT, 
                fill=0, 
                fill_mask=0
            ),
            # division scaling (matches TF image/255.0 baseline)
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            # cast to PyTorch tensors
            ToTensorV2()
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # load image and convert from default OpenCV BGR to RGB
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # load grayscale mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # apply concurrent Albumentations spatial transformations
        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed["image"]       # shape: (3, H, W)
        mask_tensor = transformed["mask"].long()   # shape: (H, W)
        
        # clamp target values to [0, NUM_CLASSES - 1]
        clamped_mask = torch.clamp(mask_tensor, 0, config.NUM_CLASSES - 1)
        
        # one-hot encoding conversion: (H, W) -> (Classes, H, W)
        # project the class dim and permute to Channel-First standard for PyTorch
        mask_onehot = torch.eye(config.NUM_CLASSES)[clamped_mask].permute(2, 0, 1)
        
        return image_tensor, mask_onehot


def get_file_pairs(directory: str) -> Tuple[List[str], List[str]]:
    """
    utility function to discover and match image-mask file pairs.
    """
    if not os.path.exists(directory):
        return [], []
    images = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".jpg")])
    masks = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith("_mask.png")])
    return images, masks


def compute_global_class_weights(mask_paths: List[str], num_classes: int) -> List[float]:
    """
    class-agnostic pre-computation tool.
    computes true inverse-frequency weights over the entire training subset.
    """
    print(f"[preprocessing] computing global class weights over {len(mask_paths)} files for {num_classes} classes...")
    class_counts = np.zeros(num_classes, dtype=np.int64)
    
    for path in mask_paths:
        mask = Image.open(path).convert('L')
        mask_np = np.array(mask)
        
        # clamp raw values to match network classes
        clamped_mask = np.clip(mask_np, 0, num_classes - 1)
        counts = np.bincount(clamped_mask.ravel(), minlength=num_classes)
        class_counts += counts
        
    total_pixels = class_counts.sum()
    
    # prevent division by zero if any rare class is absent in split
    class_counts = np.maximum(class_counts, 1)
    
    # calculate inverse frequency weights
    weights = total_pixels / (num_classes * class_counts.astype(np.float64))
    print(f" -> true global weights: {weights.tolist()}")
    return weights.tolist()


if __name__ == "__main__":
    # local module level test run
    print("[dataset test] running pipeline verification...")
    images, masks = get_file_pairs(config.TRAIN_DIR)
    
    if len(images) == 0:
        print("[dataset test WARNING] no local training images found, skipping verification.")
    else:
        dataset = DentalDataset(images[:5], masks[:5], is_training=True)
        img, msk = dataset[0]
        print(f" -> completed: sample image tensor shape: {img.shape} (expected: (3, 256, 256))")
        print(f" -> completed: sample mask tensor shape: {msk.shape} (expected: (5, 256, 256))")
        print(f" -> tensor data types: image={img.dtype}, mask={msk.dtype}")
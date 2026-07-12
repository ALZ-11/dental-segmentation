import os
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from PIL import Image

def compute_global_class_weights(mask_paths: List[str]) -> Tuple[float, float]:
    """
    iterates over the training subset of mask files to calculate 
    global inverse frequency weights for classes: [0: Background, 1: Teeth].
    """
    print(f"[preprocessing] computing global class weights over {len(mask_paths)} files...")
    
    class_counts = np.zeros(2, dtype=np.int64)
    
    for path in mask_paths:
        # load mask using PIL
        mask = Image.open(path).convert('L')
        mask_np = np.array(mask)
        
        # match original baseline: map class 4 (teeth) to 1, others to 0
        binary_mask = np.where(mask_np == 4, 1, 0)
        
        # accumulate pixel distribution
        counts = np.bincount(binary_mask.ravel(), minlength=2)
        class_counts += counts
        
    total_pixels = class_counts.sum()
    if total_pixels == 0:
        raise ValueError("calculated total pixel space is zero, check mask directory path validity.")
        
    # calculate inverse frequency weights
    # formula: total_samples / (num_classes * class_samples)
    weights = total_pixels / (2.0 * class_counts)
    print(f" -> completed... background pixels: {class_counts[0]}, teeth pixels: {class_counts[1]}")
    print(f" -> true global weights: [0: {weights[0]:.4f}, 1: {weights[1]:.4f}]")
    return float(weights[0]), float(weights[1])


def parse_image_and_mask_fn(img_path: tf.Tensor, mask_path: tf.Tensor, target_shape: Tuple[int, int] = (256, 256)):
    """
    asynchronous tensor pipeline execution function.
    reads raw bytes from disk, letterboxes preserving target aspect ratio, 
    normalizes values, converts binary classes, and one-hot encodes.
    """
    # read and decode clinical photo
    img_bytes = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(img_bytes, channels=3)
    # normalize image to [0.0, 1.0] and convert to float32
    image = tf.cast(image, tf.float32) / 255.0
    
    # read and decode grayscale label mask
    mask_bytes = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask_bytes, channels=1)
    mask = tf.cast(mask, tf.int32)
    
    # apply zero-padded letterboxing
    # bilinear interpolation for clinical features
    image_padded = tf.image.resize_with_pad(image, target_shape[0], target_shape[1], method='bilinear')
    # nearest neighbor for categorical boundaries (to prevent edge-bleeding artifacts)
    mask_padded = tf.image.resize_with_pad(mask, target_shape[0], target_shape[1], method='nearest')
    
    # binary label transformation
    # original baseline target value: 4 representing teeth
    binary_mask = tf.where(tf.equal(mask_padded, 4), tf.ones_like(mask_padded), tf.zeros_like(mask_padded))
    binary_mask = tf.squeeze(binary_mask, axis=-1) # drop dummy single channel: (H, W)
    
    # pipeline scale agnostic conversion
    mask_onehot = tf.one_hot(binary_mask, depth=2) # outputs (H, W, 2)
    
    return image_padded, mask_onehot


def get_dataset_pipeline(
    img_paths: List[str], 
    mask_paths: List[str], 
    batch_size: int = 8, 
    is_training: bool = True,
    target_shape: Tuple[int, int] = (256, 256)
) -> tf.data.Dataset:
    """
    compiles and constructs a multi-threaded, prefetching tf.data.Dataset pipeline.
    """
    # convert file lists directly to tensor slices
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    
    # shuffle only during active training runs
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(img_paths), reshuffle_each_iteration=True)
        
    # map loading & letterboxing functions in parallel on background CPU threads
    dataset = dataset.map(
        lambda img, msk: parse_image_and_mask_fn(img, msk, target_shape),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # batch formatted elements
    dataset = dataset.batch(batch_size)
    
    # prefetch batch N+1 while the GPU is executing batch N
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
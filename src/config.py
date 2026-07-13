# src/config.py
import os
import torch

# environment & hardware auto detection
IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists("/kaggle/input")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# path resolutions
if IS_KAGGLE:
    DATASET_DIR = "/kaggle/input/datasets/pawanvalluri/dental-segmentation/dentalai-2"
else:
    DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# model config (using SOTA SMP library)
MODEL_NAME = "unet"
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"
NUM_CLASSES = 5              # (updated to 5 class)
TARGET_SIZE = (256, 256)

# training hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 20
GAMMA = 0.35                 # power-law smoothing exponent for stable gradients

# logging config
print(f"[config loaded] device: {DEVICE.upper()} | local run: {not IS_KAGGLE} | dataset path: {DATASET_DIR}")
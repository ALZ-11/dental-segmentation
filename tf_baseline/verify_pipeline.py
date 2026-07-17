# tf_baseline/verify_pipeline.py
import os
from src.metrics import verify_metrics
from src.data_pipeline import parse_image_and_mask_fn

if __name__ == "__main__":
    verify_metrics()
    print("\n[verification passed] metrics component behaves as expected.")
    print("deploy codebase structure to Kaggle to compile loss function and baseline model.")
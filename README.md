# Dental Pathology Segmentation using an Optimized PyTorch Lightning U-Net

A deep learning repository designed to perform 5-class semantic segmentation on clinical dental photography, isolating healthy tooth structures from localized caries, structural anomalies, and restorations under extreme class imbalance.

## Project Overview
* **Clinical Task**: Map raw clinical photographs into 5 semantic categories: Background (0), Dental Caries/Decay (1), Structural Anomalies (2), Restorations/Fillings (3), and Healthy Teeth (4).
* **Modernized Engine**: Rebuilt from a Keras baseline to an asynchronous, pre-fetched PyTorch Lightning pipeline featuring transfer learning (U-Net with a pre-trained ResNet34 encoder).
* **Achievement**: Slashed training times by over **10x** while scaling the diagnostic output from a simple teeth-only mask to a balanced, 5-class clinical pathology classifier.

---

## Technical Performance & Progress Tracker

The development of this repository involved a transition from an unoptimized prototype to a stable, and performant framework:

| Metric | Legacy Keras Prototype (Stage 1) | Modernized PyTorch Baseline (Stage 2) | 5-Class Pathology Model (Stage 3) |
| :--- | :--- | :--- | :--- |
| **Task Scope** | Binary (Teeth vs Background) | Binary (Teeth vs Background) | **5-Class Clinical Pathology** |
| **Model Depth** | Custom 3-Stage U-Net | SMP U-Net (ResNet34 backbone) | **SMP U-Net (ResNet34 backbone)** |
| **Epoch Duration** | ~600–700 seconds (CPU) | **~61 seconds** (10x faster) | **~58–60 seconds** |
| **Best Validation Loss** | `0.1711` (Standard Cross Entropy) | `0.1550` (Joint Dice + CE Loss) | **`0.6712`** (Weighted Joint Loss) |
| **Macro Jaccard IoU** | **`0.7690` (76.90%)** | **`0.9040` (90.40%)** | **`0.4790` (47.90%)** |
| **Unseen Test IoU** | — | — | **`0.4402` (44.02%)** |

---

## Architectural & Engineering Highlights

### 1. Aspect-Ratio Preserving Preprocessing
Simple vertical and horizontal squishing distorts teeth geometries, degrading a model's ability to isolate fine pathological details. This repository implements a zero-padded letterboxing transform using the **Albumentations** library:
* **`A.LongestMaxSize(max_size=256)`**: Rescales the input photo so that its longest dimension is 256 pixels, preserving raw anatomical proportions.
* **`A.PadIfNeeded`**: Adds zero-padding (black borders) symmetrically to the shorter side, outputting a non-distorted $256 \times 256$ tensor.

### 2. Numerical Gradient Stabilization (Power-Law Weight Smoothing)
In the raw 5-class clinical dataset, background and healthy teeth take up >99.9% of the pixel space, leaving caries and restorations highly sparse. Standard inverse-frequency weights create massive loss spikes ($W_c$ range: $[0.23, \dots, 1274.53]$), trapping the optimizer in local minima almost immediately.
A configurable **power-law smoothing factor ($\gamma = 0.35$)** is implemented inside the weight calculator to compress the penalty range:
$$W_c = \left( \frac{\text{Total Pixels}}{\text{Pixels}_c} \right)^\gamma$$
This compresses weights to a stable scale ($[1.0, \dots, 20.16]$), which preserves heavy minority-class penalties while ensuring stable gradient updates.

### 3. Multi-Class Loss Function
A unified, joint loss function combining pixel-wise classification confidence with geometric overlap optimization is utilized:
$$\text{Joint Loss} = \text{nn.CrossEntropyLoss}(\text{weight}=\mathbf{W}) + \text{smp.losses.DiceLoss}(\text{mode="multiclass"})$$

---

## Repository Structure
```text
dental-segmentation/
├── dataset/                  # gitignored (Place raw Kaggle folds here)
│   ├── train/
│   ├── valid/
│   └── test/
├── notebooks/
│   └── dental_seg_SMPtraining.ipynb  # Kaggle GPU orchestrator
├── src/                      # PyTorch root files
│   ├── config.py             # Centralized parameters and auto-detect paths
│   ├── dataset.py            # PyTorch Dataset & Albumentations letterboxing
│   ├── model.py              # PyTorch Lightning module (SMP, metrics, Joint Loss)
│   └── train.py              # Main training pipeline executor
├── tf_baseline/              # Archived legacy TensorFlow implementation
│   ├── verify_pipeline.py    # Baseline verification utility
│   ├── notebooks/
│   │   ├── dental_seg_prototype.ipynb  # The original Keras prototype
│   │   └── dental_seg_rectifiedIoU.ipynb
│   └── src/
│       └── ... (Legacy TF modules)
├── .gitignore
└── requirements.txt
```

---

## Getting Started

### 1. Installation
Clone the repository and install PyTorch, Lightning, and Albumentations dependencies:
```bash
git clone https://github.com/ALZ-11/dental-segmentation
cd dental-segmentation
pip install -r requirements.txt
```

### 2. Dataset Setup
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/pawanvalluri/dental-segmentation) and extract the folders into the `/dataset` root directory:
```text
/dataset/train
/dataset/valid
/dataset/test
```

### 3. Execution
The repository features automated environment detection. It automatically checks if it is executing locally or in a Kaggle container by resolving paths and accelerators (GPU vs CPU).

* **To train the model locally**:
  ```bash
  python src/train.py
  ```
* **To check module-level integrity locally**:
  ```bash
  python -m src.dataset
  python -m src.model
  ```
* **To run training on Kaggle (recommended)**: 
  Simply upload and execute the cells inside `notebooks/dental_seg_SMPtraining.ipynb`. The notebook contains Python setups that protect against pathing and cloning errors. Make sure to enable acceleration and internet access on Kaggle.

---

## Evaluation & Diagnostic Plotting

Executing the final evaluation block on unseen test photos outputs a 3-column comparative plot. Tensors are mapped to a color palette:

* **Class 0 (Background)**: Black
* **Class 1 (Caries/Decay)**: Red
* **Class 2 (Anomalies)**: Yellow
* **Class 3 (Restorations/Fillings)**: Blue
* **Class 4 (Healthy Teeth)**: Green

Evaluations on the unseen test split sometimes reveal out-of-distribution boundary detection, successfully identifying active carious lesions even in cases where the original ground-truth annotators missed them (label noise).
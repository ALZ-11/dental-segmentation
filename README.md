# High-Fidelity Dental Image Segmentation using an Optimized 3-Stage U-Net

An optimized deep learning repository designed to perform binary semantic segmentation on clinical dental photography, isolating teeth structures from background, soft tissue, and orthodontic appliances.

## Project Overview
* **Clinical Task**: Isolate teeth contours (Class 1) from background and clinical oral environments (Class 0).
* **Architecture**: Optimized 3-stage deep U-Net containing Batch Normalization (490,130 parameters, ~1.87 MB).
* **Key Results**: **91.40% Test Accuracy** with a **0.1870 Test Loss** on unseen clinical photography.
* **Optimization Highlights**: Custom Weighted Categorical Cross-Entropy, dynamic epoch-level data generator shuffling, and custom class weight balancing.

---

## Technical Performance & Metrics

### Training Parameters
* **Target Resolution**: 256x256 pixels (Bilinear interpolation for images, Nearest-Neighbor for masks)
* **Optimizer**: Adam (Learning Rate: 1e-4)
* **Callbacks**: `ModelCheckpoint` monitoring `val_loss`, `EarlyStopping` (Patience: 5, restoring optimal weights)
* **Hardware Profile**: CPU-bound execution (optimized via Batch Normalization layers)

### Performance Curves
Training terminated at Epoch 20, with Keras restoring the optimal weights from **Epoch 19** where validation loss hit a minimal value of **0.1757** with a **91.24% Validation Accuracy**.

* **Test Loss**: 0.1870
* **Test Accuracy**: 91.40%

---

## Repository Structure
```text
dental-segmentation/
├── dataset/                  # Ignored by Git (Place raw folders here)
│   ├── train/
│   ├── valid/
│   └── test/
├── models/
│   └── best_model.keras      # Pretrained 3-Stage U-Net weights (~1.8 MB)
├── notebooks/
│   └── dental_segmentation_prototype.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Installation
Clone the repository and install dependencies:
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

### 3. Running Inference
Open the Jupyter Notebook inside the `/notebooks` folder and execute the cells. The pipeline automatically falls back to your local path and loads the pre-trained weights from `models/best_model.keras` to visualize the predictions instantly.

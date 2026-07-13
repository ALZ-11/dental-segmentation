# src/train.py
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from src import config
from src.dataset import DentalDataset, get_file_pairs
from src.model import DentalSegmentationModel

def train_pipeline():
    print("[training pipeline] starting pipeline configuration...")
    
    # discover file pairs for dataset splits
    train_images, train_masks = get_file_pairs(config.TRAIN_DIR)
    valid_images, valid_masks = get_file_pairs(config.VALID_DIR)
    
    print(f" -> found {len(train_images)} training images, {len(valid_images)} validation images.")
    
    if len(train_images) == 0:
        print("[abort WARNING] no dataset directories found. verification run ended.")
        return
        
    # instantiate dataset modules
    train_dataset = DentalDataset(train_images, train_masks, is_training=True)
    valid_dataset = DentalDataset(valid_images, valid_masks, is_training=False)
    
    # configure multi-threaded & memory-pinned PyTorch DataLoaders
    # num_workers are allocated based on available CPU cores (prevent CPU-GPU bottlenecking)
    num_workers = os.cpu_count() if os.cpu_count() else 2
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == "cuda" else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == "cuda" else False
    )
    
    model = DentalSegmentationModel()
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="best_model_sota-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        verbose=True
    )
    
    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        accelerator="gpu" if config.DEVICE == "cuda" else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10
    )
    
    print("[training pipeline] handing execution to PyTorch Lightning trainer...")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    train_pipeline()
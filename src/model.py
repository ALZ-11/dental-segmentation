# src/model.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassJaccardIndex
from src import config

class DentalSegmentationModel(pl.LightningModule):
    """
    PyTorch Lightning Segmentation Module.
    encapsulates SMP U-Net, joint loss (CE + Dice), and performance metrics.
    """
    def __init__(self):
        super().__init__()
        # instantiate SMP U-Net model using pre-trained ResNet34 encoder
        self.model = smp.Unet(
            encoder_name=config.ENCODER_NAME,
            encoder_weights=config.ENCODER_WEIGHTS,
            in_channels=3,
            classes=config.NUM_CLASSES
        )
        
        # define joint loss components: soft cross entropy + overlap dice loss
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass")
        self.ce_loss = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.0)
        
        # configure performance metrics (automatically handles Jaccard index argmax checks)
        self.train_iou = MulticlassJaccardIndex(num_classes=config.NUM_CLASSES)
        self.val_iou = MulticlassJaccardIndex(num_classes=config.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch # x: (Batch, 3, 256, 256), y: (Batch, 2, 256, 256)
        y_hat = self(x) # logits: (Batch, 2, 256, 256)
        
        # extract 3D class indices shape (Batch, H, W) for SMP multiclass losses (fix)
        y_idx = torch.argmax(y, dim=1).long()
        
        # compute joint loss
        loss_dice = self.dice_loss(y_hat, y_idx)
        loss_ce = self.ce_loss(y_hat, y_idx)
        loss = loss_dice + loss_ce
        
        # compute metrics (extract argmax to map probabilities to discrete class indexes)
        y_hat_idx = torch.argmax(y_hat, dim=1)
        iou = self.train_iou(y_hat_idx, y_idx)
        
        # log training step parameters
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_iou", iou, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        
        # extract 3D class indices shape (Batch, H, W) for SMP multiclass losses (fix)
        y_idx = torch.argmax(y, dim=1).long()
        
        # compute joint loss
        loss_dice = self.dice_loss(y_hat, y_idx)
        loss_ce = self.ce_loss(y_hat, y_idx)
        loss = loss_dice + loss_ce
        
        # compute metrics
        y_hat_idx = torch.argmax(y_hat, dim=1)
        iou = self.val_iou(y_hat_idx, y_idx)
        
        # log validation step parameters
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_iou", iou, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        # configure Adam optimizer using config parameters
        return torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE)


if __name__ == "__main__":
    # local module check
    print("[model test] initializing DentalSegmentationModel...")
    model = DentalSegmentationModel()
    
    # generate a dummy batch (representing 1 clinical image of size 256x256)
    dummy_batch = torch.randn(1, 3, 256, 256)
    output = model(dummy_batch)
    
    print(f" -> completed: model output prediction shape: {output.shape} (expected: (1, 2, 256, 256))")
    print("[model test passed] compile successful.")
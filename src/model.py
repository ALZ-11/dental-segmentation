# src/model.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassJaccardIndex
try:
    from src import config
except:
    import config

class DentalSegmentationModel(pl.LightningModule):
    """
    PyTorch Lightning Segmentation Module.
    encapsulates SMP U-Net, joint loss (CE + Dice), and performance metrics.
    """
    def __init__(self, class_weights: list = None):
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
        
        # if class weights are provided, compile a weighted CrossEntropyLoss
        if class_weights is not None:
            self.register_buffer("class_weights_tensor", torch.tensor(class_weights, dtype=torch.float32))
            self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        # configure performance metrics (automatically handles Jaccard index argmax checks)
        self.train_iou = MulticlassJaccardIndex(num_classes=config.NUM_CLASSES)
        self.val_iou = MulticlassJaccardIndex(num_classes=config.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch # x: (Batch, 3, 256, 256), y: (Batch, 5, 256, 256)
        y_hat = self(x) # logits: (Batch, 5, 256, 256)
        
        # extract 3D class indices shape (Batch, H, W) for standard losses
        y_idx = torch.argmax(y, dim=1).long()
        
        # compute joint loss
        loss_dice = self.dice_loss(y_hat, y_idx)
        loss_ce = self.ce_loss(y_hat, y_idx)
        loss = loss_dice + loss_ce
        
        # compute metrics
        y_hat_idx = torch.argmax(y_hat, dim=1)
        iou = self.train_iou(y_hat_idx, y_idx)
        
        # log training step parameters
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_iou", iou, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        
        # extract 3D class indices shape (Batch, H, W) for standard losses
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
    model = DentalSegmentationModel(class_weights=[0.1, 1.5, 2.0, 1.0, 0.2])
    
    # generate a dummy batch (representing 1 clinical image of size 256x256)
    dummy_batch = torch.randn(1, 3, 256, 256)
    output = model(dummy_batch)
    
    print(f" -> completed: model output prediction shape: {output.shape} (expected: (1, 5, 256, 256))")
    print("[model test passed] compile successful.")
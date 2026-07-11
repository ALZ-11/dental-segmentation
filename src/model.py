from typing import Tuple
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, concatenate, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Model

def unet_model(input_shape: Tuple[int, int, int] = (256, 256, 3), num_classes: int = 2) -> Model:
    """
    instantiates a lightweight 3-stage custom U-Net CNN.
    parameter count: 490,130 (~1.87 MB)
    """
    inputs = Input(shape=input_shape, name="input_layer")
    
    # --- Encoder (Downsampling) ---
    # Level 1 (256x256 -> 128x128)
    c1 = Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPool2D((2, 2))(c1)
    
    # Level 2 (128x128 -> 64x64)
    c2 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPool2D((2, 2))(c2)
    
    # Level 3 (64x64 -> 32x32)
    c3 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.1)(c3)
    c3 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPool2D((2, 2))(c3)
    
    # --- Bottleneck (32x32) ---
    c4 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
    c4 = BatchNormalization()(c4)
    
    # --- Decoder (Upsampling) ---
    # Level 3 (32x32 -> 64x64)
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u5)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.1)(c5)
    c5 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)
    c5 = BatchNormalization()(c5)
    
    # Level 2 (64x64 -> 128x128)
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.1)(c6)
    c6 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)
    c6 = BatchNormalization()(c6)
    
    # Level 1 (128x128 -> 256x256)
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.1)(c7)
    c7 = Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)
    c7 = BatchNormalization()(c7)
    
    # --- Output Layer ---
    outputs = Conv2D(num_classes, (1, 1), activation="softmax", name="output_layer")(c7)
    
    model = Model(inputs=inputs, outputs=outputs, name="functional_unet")
    return model

if __name__ == "__main__":
    print("compiling U-Net model...")
    model = unet_model(input_shape=(256, 256, 3), num_classes=2)
    model.summary()
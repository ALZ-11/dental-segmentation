import tensorflow as tf
from typing import List, Union
import numpy as np

def weighted_categorical_crossentropy(weights: Union[List[float], np.ndarray]):
    """
    factory function returning a weighted categorical 
    cross-entropy loss function.
    
    Args:
        weights: list or array of class weights to scale channel losses.
    """
    weights_tensor = tf.convert_to_tensor(list(weights), dtype=tf.float32)
    
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # cast tensors for floating-point calculation consistency
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # clip predictions to prevent absolute zeros/ones (prevents log(0) NaN instability)
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # calculate cross-entropy
        ce = y_true * tf.math.log(y_pred)
        
        # scale loss channels by their respective weights
        weighted_ce = -tf.reduce_sum(ce * weights_tensor, axis=-1)
        
        return tf.reduce_mean(weighted_ce)
        
    return loss
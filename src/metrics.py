import tensorflow as tf

class ArgmaxMeanIoU(tf.keras.metrics.MeanIoU):
    """
    optimized Mean IoU metric wrapper.
    converts one-hot encoded probabilities into discrete integer class 
    indices using argmax before evaluating the confusion matrix.
    """
    def __init__(self, num_classes: int, name: str = 'mean_io_u', dtype=None):
        super().__init__(num_classes=num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # cast inputs if needed and compute argmax along the class channel
        y_true_idx = tf.argmax(y_true, axis=-1)
        y_pred_idx = tf.argmax(y_pred, axis=-1)
        
        # pass integer class maps to the parent Keras MeanIoU implementation
        return super().update_state(y_true_idx, y_pred_idx, sample_weight)


def verify_metrics():
    """
    unit test bed to verify metric calculation behavior.
    """
    print("[metric test] initializing ArgmaxMeanIoU self-test...")
    metric = ArgmaxMeanIoU(num_classes=2)

    # ground truth: shape (Batch=1, H=2, W=2, Classes=2)
    y_true = tf.constant([[[[1., 0.], [0., 1.]], 
                           [[1., 0.], [1., 0.]]]])

    # 1. test case: perfect match
    y_pred_perfect = y_true
    metric.update_state(y_true, y_pred_perfect)
    perfect_result = metric.result().numpy()
    print(f" -> test 1 (perfect match) IoU: {perfect_result:.4f} (expected: 1.0000)")
    metric.reset_state()

    # 2. test case: completely inverted match (opposite probabilities)
    y_pred_inverted = 1.0 - y_true
    metric.update_state(y_true, y_pred_inverted)
    inverted_result = metric.result().numpy()
    print(f" -> test 2 (completely inverted) IoU: {inverted_result:.4f} (expected: 0.0000)")
    metric.reset_state()

    # 3. test case: class 0 (background) only prediction everywhere
    y_pred_bg = tf.constant([[[[1., 0.], [1., 0.]], 
                             [[1., 0.], [1., 0.]]]])
    metric.update_state(y_true, y_pred_bg)
    bg_result = metric.result().numpy()
    # class 0: true pos = 3, pred pos = 4 -> union = 4, intersect = 3. IoU_0 = 3/4 = 0.75
    # class 1: true pos = 1, pred pos = 0 -> union = 1, intersect = 0. IoU_1 = 0.0
    # mean IoU = (0.75 + 0) / 2 = 0.375
    print(f" -> test 3 (all background prediction) IoU: {bg_result:.4f} (expected: 0.3750)")
    metric.reset_state()


if __name__ == "__main__":
    verify_metrics()
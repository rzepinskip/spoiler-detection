
import tensorflow as tf
import tensorflow_addons as tfa


# Adapted from https://www.tensorflow.org/guide/keras/train_and_evaluate#specifying_a_loss_metrics_and_an_optimizer
class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    """
    Args:
      pos_weight: Scalar to affect the positive labels of the loss function.
      weight: Scalar to affect the entirety of the loss function.
      from_logits: Whether to compute loss from logits or the probability.
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """
    def __init__(self, pos_weight=1, weight=1, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='weighted_binary_crossentropy'):
        super().__init__(reduction=reduction, name=name)
        self.pos_weight = pos_weight
        self.weight = weight
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        ce = tf.losses.binary_crossentropy(
            y_true, y_pred, from_logits=self.from_logits)[:,None]
        ce = self.weight * (ce*(1-y_true) + self.pos_weight*ce*(y_true))
        return ce


class SscAuc(tf.keras.metrics.AUC):
    def __init__(self, **kwargs):
        super(SscAuc, self).__init__(name="auc", **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.not_equal(y_true, -1)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        super(SscAuc, self).update_state(y_true, y_pred, sample_weight)


class SscWeightedBinaryCrossEntropy(WeightedBinaryCrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        mask = tf.not_equal(y_true, -1)
        y_true = tf.expand_dims(tf.boolean_mask(y_true, mask), 1)
        y_pred = tf.expand_dims(tf.boolean_mask(y_pred, mask), 1)
        return super(SscWeightedBinaryCrossEntropy, self).call(y_true, y_pred)

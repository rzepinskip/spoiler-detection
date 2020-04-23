
import tensorflow as tf
import tensorflow_addons as tfa


class SscAuc(tf.keras.metrics.AUC):
    def __init__(self, **kwargs):
        super(SscAuc, self).__init__(name="auc", **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.slice(tf.reshape(y_true, [-1]), [0], [y_pred.shape[1]])
        super(SscAuc, self).update_state(y_true, y_pred, sample_weight)


class SscBinaryCrossEntropy(tf.keras.losses.BinaryCrossentropy):
    def __init__(self, **kwargs):
        super(SscBinaryCrossEntropy, self).__init__(name="loss", **kwargs)

    def call(self, y_true, y_pred):
        y_true = tf.slice(tf.reshape(y_true, [-1]), [0], [y_pred.shape[1]])
        return super(SscBinaryCrossEntropy, self).call(y_true, y_pred)

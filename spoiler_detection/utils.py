import tensorflow as tf
from tensorflow.keras import backend as K





def focal_loss(gamma=2.0, alpha=0.2):
    # https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1.0 - pt_1, gamma) * K.log(pt_1)) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1.0 - pt_0)
        )

    return focal_loss_fixed


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

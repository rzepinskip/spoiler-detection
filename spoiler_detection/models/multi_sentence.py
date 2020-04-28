import tensorflow as tf
import transformers
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.mixed_precision.experimental import (
    loss_scale_optimizer as lso,
)


# Adapted from https://github.com/tensorflow/tensorflow/blob/1381fc8e15e22402417b98e3881dfd409998daea/tensorflow/python/keras/engine/training.py#L540
class SscModel(tf.keras.Model):
    def __init__(self, hparams):
        super(SscModel, self).__init__()
        self.transformer = transformers.TFAutoModel.from_pretrained(hparams.model_type)
        self.dropout = tf.keras.layers.Dropout(hparams.dropout)
        self.classifier = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, **kwargs):
        input_ids = inputs
        attention_mask = tf.where(
            tf.equal(inputs, 0), tf.zeros_like(inputs), tf.ones_like(inputs)
        )
        sequence_output = self.transformer([input_ids, attention_mask], **kwargs)[0]
        x = self.dropout(sequence_output, training=kwargs.get("training", False))
        out = self.classifier(x)
        return out

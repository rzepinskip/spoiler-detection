import tensorflow as tf
import transformers


class SequenceModel(tf.keras.Model):
    def __init__(self, hparams):
        super(SequenceModel, self).__init__()
        self.transformer = transformers.TFAutoModel.from_pretrained(hparams.model_type)
        self.dropout = tf.keras.layers.Dropout(hparams.dropout)
        self.classifier = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        sequence_output = self.transformer(inputs)[0]
        cls_token = sequence_output[:, 0, :]
        x = self.dropout(cls_token)
        out = self.classifier(x)
        return out


class PooledModel(tf.keras.Model):
    def __init__(self, hparams):
        super(PooledModel, self).__init__()
        self.transformer = transformers.TFAutoModel.from_pretrained(hparams.model_type)
        self.dropout = tf.keras.layers.Dropout(hparams.dropout)
        self.classifier = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        pooled_output = self.transformer(inputs)[1]
        x = self.dropout(pooled_output)
        out = self.classifier(x)
        return out

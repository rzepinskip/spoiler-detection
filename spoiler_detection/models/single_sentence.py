import tensorflow as tf
import transformers


class SequenceModel(tf.keras.Model):
    def __init__(self, hparams):
        super(SequenceModel, self).__init__()
        self.transformer = transformers.TFAutoModel.from_pretrained(hparams.model_type)
        self.dropout = tf.keras.layers.Dropout(hparams.dropout)
        self.classifier = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, **kwargs):
        input_ids = inputs
        attention_mask = tf.where(
            tf.equal(inputs, 0), tf.zeros_like(inputs), tf.ones_like(inputs)
        )
        sequence_output = self.transformer([input_ids, attention_mask], **kwargs)[0]
        cls_token = sequence_output[:, 0, :]
        x = self.dropout(cls_token, training=kwargs.get("training", False))
        out = self.classifier(x)
        return out


class PooledModel(tf.keras.Model):
    def __init__(self, hparams):
        super(PooledModel, self).__init__()
        self.transformer = transformers.TFAutoModel.from_pretrained(hparams.model_type)
        self.dropout = tf.keras.layers.Dropout(hparams.dropout)
        self.classifier = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, **kwargs):
        input_ids = inputs
        attention_mask = tf.where(
            tf.equal(inputs, 0), tf.zeros_like(inputs), tf.ones_like(inputs)
        )
        pooled_output = self.transformer([input_ids, attention_mask], **kwargs)[1]
        x = self.dropout(pooled_output, training=kwargs.get("training", False))
        out = self.classifier(x)
        return out

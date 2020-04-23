import tensorflow as tf
import transformers


class SequenceModel(tf.keras.Model):
    def __init__(self, hparams):
        super(SequenceModel, self).__init__()
        self.transformer = transformers.TFAutoModel.from_pretrained(hparams.model_type)
        self.dropout = tf.keras.layers.Dropout(hparams.dropout)
        self.classifier = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        input_dict = {
            "input_ids": inputs,
            "token_type_ids": tf.zeros_like(inputs),
            "attention_mask": tf.where(
                tf.equal(inputs, 0), tf.zeros_like(inputs), tf.ones_like(inputs)
            ),
        }
        x = self.transformer(input_dict, training=training)[0]
        x = x[:, 0, :]
        if training:
            x = self.dropout(x, training=training)
        x = self.classifier(x)
        return x


class PooledModel(tf.keras.Model):
    def __init__(self, hparams):
        super(PooledModel, self).__init__()
        self.transformer = transformers.TFAutoModel.from_pretrained(hparams.model_type)
        self.dropout = tf.keras.layers.Dropout(hparams.dropout)
        self.classifier = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        input_dict = {
            "input_ids": inputs,
            "token_type_ids": tf.zeros_like(inputs),
            "attention_mask": tf.where(
                tf.equal(inputs, 0), tf.zeros_like(inputs), tf.ones_like(inputs)
            ),
        }
        x = self.transformer(input_dict, training=training)[1]
        if training:
            x = self.dropout(x, training=training)
        x = self.classifier(x)
        return x

import tensorflow as tf
import transformers


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
        sep_mask = inputs == 102  # TODO unhardcode
        sep_embeddings = sequence_output[sep_mask]
        x = self.dropout(sep_embeddings, training=kwargs.get("training", False))
        out = self.classifier(x)
        return out

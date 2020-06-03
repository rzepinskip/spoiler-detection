import tensorflow as tf
import transformers


def get_model(model_type):
    model = transformers.TFElectraModel.from_pretrained(
        model_type, output_attentions=True
    )

    return model


class SequenceModel(tf.keras.Model):
    def __init__(self, hparams, output_bias=None):
        super(SequenceModel, self).__init__()
        self.transformer = get_model(hparams.model_type)
        self.dropout = tf.keras.layers.Dropout(hparams.dropout)
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        self.classifier = tf.keras.layers.Dense(
            1, activation="sigmoid", bias_initializer=output_bias, name="final"
        )
        self.use_genres = hparams.use_genres
        if hparams.use_genres:
            self.genres_layer = tf.keras.layers.Dense(
                10, activation="relu", name="genre"
            )

    def call(self, inputs, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = tf.where(
            tf.equal(input_ids, 0), tf.zeros_like(input_ids), tf.ones_like(input_ids)
        )
        sequence_output = self.transformer([input_ids, attention_mask], **kwargs)[0]
        x = sequence_output[:, 0, :]  # cls_token

        if self.use_genres:
            genres = inputs["genres"]
            genres_output = self.genres_layer(genres)
            x = tf.concat([x, genres_output], -1)

        x = self.dropout(x, training=kwargs.get("training", False))
        x = self.classifier(x)
        return x

    def get_attention(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = tf.where(
            tf.equal(input_ids, 0), tf.zeros_like(input_ids), tf.ones_like(input_ids)
        )
        output = self.transformer([input_ids, attention_mask], training=False)
        return output[-1]


class PooledModel(tf.keras.Model):
    def __init__(self, hparams, output_bias=None):
        super(PooledModel, self).__init__()
        self.transformer = get_model(hparams.model_type)
        self.dropout = tf.keras.layers.Dropout(hparams.dropout)
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        self.classifier = tf.keras.layers.Dense(
            1, activation="sigmoid", bias_initializer=output_bias, name="final"
        )
        self.use_genres = hparams.use_genres
        if hparams.use_genres:
            self.genres_layer = tf.keras.layers.Dense(
                10, activation="relu", name="genre"
            )

    def call(self, inputs, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = tf.where(
            tf.equal(input_ids, 0), tf.zeros_like(input_ids), tf.ones_like(input_ids)
        )
        pooled_output = self.transformer([input_ids, attention_mask], **kwargs)[1]
        x = pooled_output

        if self.use_genres:
            genres = inputs["genres"]
            genres_output = self.genres_layer(genres)
            x = tf.concat([x, genres_output], -1)

        x = self.dropout(x, training=kwargs.get("training", False))
        x = self.classifier(x)
        return x

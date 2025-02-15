import tensorflow as tf
import transformers
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.mixed_precision.experimental import (
    loss_scale_optimizer as lso,
)


def get_model(model_type):
    if "electra" in model_type:
        return transformers.TFElectraModel.from_pretrained(model_type)

    return transformers.TFAutoModel.from_pretrained(model_type)


# Adapted from https://github.com/tensorflow/tensorflow/blob/1381fc8e15e22402417b98e3881dfd409998daea/tensorflow/python/keras/engine/training.py#L540
class SscModel(tf.keras.Model):
    def __init__(self, hparams, output_bias=None):
        super(SscModel, self).__init__()
        self.transformer = get_model(hparams.model_type)
        self.dropout = tf.keras.layers.Dropout(hparams.dropout)
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        self.classifier = tf.keras.layers.Dense(
            1, activation="sigmoid", bias_initializer=output_bias, name="final"
        )

    def call(self, inputs, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = tf.where(
            tf.equal(input_ids, 0), tf.zeros_like(input_ids), tf.ones_like(input_ids)
        )
        sequence_output = self.transformer([input_ids, attention_mask], **kwargs)[0]
        x = self.dropout(sequence_output, training=kwargs.get("training", False))
        out = self.classifier(x)
        return out

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)

            mask = tf.not_equal(y, -1)
            y = tf.expand_dims(tf.boolean_mask(y, mask), 1)
            y_pred = tf.boolean_mask(y_pred, mask)

            sample_weight = tf.ones_like(y_pred)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses
            )

        _minimize(
            self.distribute_strategy,
            tape,
            self.optimizer,
            loss,
            self.trainable_variables,
        )

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        y_pred = self(x, training=False)
        mask = tf.not_equal(y, -1)
        y = tf.expand_dims(tf.boolean_mask(y, mask), 1)
        y_pred = tf.boolean_mask(y_pred, mask)
        self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}


def _minimize(strategy, tape, optimizer, loss, trainable_variables):
    with tape:
        if isinstance(optimizer, lso.LossScaleOptimizer):
            loss = optimizer.get_scaled_loss(loss)

    gradients = tape.gradient(loss, trainable_variables)

    aggregate_grads_outside_optimizer = (
        optimizer._HAS_AGGREGATE_GRAD
        and not isinstance(  # pylint: disable=protected-access
            strategy.extended, parameter_server_strategy.ParameterServerStrategyExtended
        )
    )

    if aggregate_grads_outside_optimizer:
        gradients = optimizer._aggregate_gradients(
            zip(gradients, trainable_variables)  # pylint: disable=protected-access
        )
    if isinstance(optimizer, lso.LossScaleOptimizer):
        gradients = optimizer.get_unscaled_gradients(gradients)
    gradients = optimizer._clip_gradients(gradients)  # pylint: disable=protected-access
    if trainable_variables:
        if aggregate_grads_outside_optimizer:
            optimizer.apply_gradients(
                zip(gradients, trainable_variables),
                experimental_aggregate_gradients=False,
            )
        else:
            optimizer.apply_gradients(zip(gradients, trainable_variables))

import gzip
import hashlib
import json
import math
import os
import re
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import transformers
import wandb
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
from wandb.keras import WandbCallback

from spoiler_detection.datasets.goodreads import get_goodreads_single, get_goodreads_ssc
from spoiler_detection.optimization import create_optimizer
from spoiler_detection.utils import SscAuc, SscBinaryCrossEntropy

tf.random.set_seed(44)
np.random.seed(44)
warnings.simplefilter("ignore")

MODEL_TYPE = "bert-base-uncased"
EPOCHS = 4
BATCH_SIZE = 2
LR = 1e-6
MAX_LENGTH = 128

AUTO = tf.data.experimental.AUTOTUNE

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print("Running on TPU")
except ValueError:
    tpu = None
strategy = tf.distribute.get_strategy()
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)

"""## Load text data into memory"""

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_TYPE, use_fast=True)

train_path = "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-timings-train.json.gz"
val_path = "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-timings-val.json.gz"
# train_path = "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-train.json.gz"
# val_path = "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-val.json.gz"
train_dataset_raw, y_train = get_goodreads_single(train_path, tokenizer, MAX_LENGTH,)
val_dataset_raw, _ = get_goodreads_single(val_path, tokenizer, MAX_LENGTH,)

train_dataset = train_dataset_raw.shuffle(2048).batch(BATCH_SIZE).prefetch(AUTO)
val_dataset = val_dataset_raw.batch(BATCH_SIZE).cache().prefetch(AUTO)


class SscModel(tf.keras.Model):
    def __init__(self, transformer):
        super(SscModel, self).__init__()
        self.transformer = transformer
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.classifier = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        sequence_output = self.transformer(inputs)[0]
        sep_mask = inputs == 102
        sep_embeddings = sequence_output[sep_mask]
        x = self.dropout(sep_embeddings)
        out = self.classifier(x)
        return out


class SequenceSingleModel(tf.keras.Model):
    def __init__(self, transformer):
        super(SequenceSingleModel, self).__init__()
        self.transformer = transformer
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.classifier = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        sequence_output = self.transformer(inputs)[0]
        cls_token = sequence_output[:, 0, :]
        x = self.dropout(cls_token)
        out = self.classifier(x)
        return out


class PooledSingleModel(tf.keras.Model):
    def __init__(self, transformer):
        super(PooledSingleModel, self).__init__()
        self.transformer = transformer
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.classifier = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        pooled_output = self.transformer(inputs)[1]
        x = self.dropout(pooled_output)
        out = self.classifier(x)
        return out


num_train_steps = math.ceil(len(y_train) / BATCH_SIZE) * EPOCHS

with strategy.scope():
    transformer = transformers.TFAutoModel.from_pretrained(MODEL_TYPE)
    model = SequenceSingleModel(transformer)
    optimizer = create_optimizer(LR, num_train_steps, 0.1 * num_train_steps)
    # model.compile(optimizer, loss=SscBinaryCrossEntropy(name="loss"), metrics=[SscAuc(name="auc")])
    model.compile(
        optimizer,
        # loss=tfa.losses.SigmoidFocalCrossEntropy(
        #     name="loss", reduction=tf.keras.losses.Reduction.AUTO
        # ),
        loss=tf.keras.losses.BinaryCrossentropy(name="loss"),
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            tfa.metrics.F1Score(num_classes=1, name="f1", threshold=0.5),
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
        ],
    )
    if tpu is None:
        model.run_eagerly = True

# model.summary()


class LogLearningRate(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        lr = self.model.optimizer._decayed_lr("float32").numpy()
        wandb.log({"lr": lr})


callbacks = list()
if False:
    wandb.init(
        project="spoiler_detection-keras",
        tags=[],
        config={
            "learning_rate": LR,
            "model_type": MODEL_TYPE,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
        },
    )

    wandb_cb = WandbCallback(monitor="val_auc", save_model=False)
    log_lr_cb = LogLearningRate()
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=False,
    )
    checkpoints_cb = tf.keras.callbacks.ModelCheckpoint(
        wandb.run.dir, monitor="val_loss", save_best_only=True, save_weights_only=False,
    )

    callbacks = [wandb_cb, log_lr_cb, early_stopping_cb]

negative_count, positive_count = len(y_train[y_train == 0]), len(y_train[y_train == 1])
weights = {
    0: max(negative_count, positive_count) / negative_count,
    1: max(negative_count, positive_count) / positive_count,
}

train_history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    steps_per_epoch=4,
    validation_steps=2,
    callbacks=callbacks,
    epochs=2,
    class_weight=weights,
)
# train_history = model.fit(
#     train_dataset,
#     validation_data=val_dataset,
#     callbacks=callbacks,
#     epochs=EPOCHS,
#     class_weight=weights,
# )

import math
import os
import re
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
import wandb
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
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
MAX_LENGTH = 256


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


X_train_raw, y_train = get_goodreads_single(
    "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-timings-train.json.gz"
)
X_val_raw, y_val = get_goodreads_single(
    "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-timings-val.json.gz"
)

# X_train_raw, y_train = get_goodreads_single("https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-train.json.gz")
# X_val_raw, y_val = get_goodreads_single("https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-val.json.gz")


def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []

    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i : i + chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])

    return np.array(all_ids)


# def regular_encode(texts, tokenizer, maxlen=512):
#     enc_di = tokenizer.batch_encode_plus(
#         texts,
#         return_attention_masks=False,
#         return_token_type_ids=False,
#         pad_to_max_length=True,
#         max_length=maxlen,
#     )

#     return np.array(enc_di["input_ids"])


# First load the real tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_TYPE)

# Save the loaded tokenizer locally
save_path = f"bert-base-uncased"
if not os.path.exists(save_path):
    os.makedirs(save_path)
tokenizer.save_pretrained(save_path)

fast_tokenizer = BertWordPieceTokenizer(f"bert-base-uncased/vocab.txt", lowercase=True)

X_train = fast_encode(X_train_raw, fast_tokenizer, maxlen=MAX_LENGTH)
X_val = fast_encode(X_val_raw, fast_tokenizer, maxlen=MAX_LENGTH)


"""## Build datasets objects"""

train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)


class SscModel(tf.keras.Model):
    def __init__(self, transformer):
        super(SscModel, self).__init__()
        self.transformer = transformer
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.classifier = Dense(1, activation="sigmoid")

    def call(self, inputs):
        sequence_output = self.transformer(inputs)[0]
        sep_mask = inputs == 102
        sep_embeddings = sequence_output[sep_mask]
        # cls_token = sequence_output[:, 0, :]
        # x = self.dropout(cls_token)
        x = self.dropout(sep_embeddings)
        out = self.classifier(x)
        return out


class SingleModel(tf.keras.Model):
    def __init__(self, transformer):
        super(SingleModel, self).__init__()
        self.transformer = transformer
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.classifier = Dense(1, activation="sigmoid")

    def call(self, inputs):
        sequence_output = self.transformer(inputs)[0]
        cls_token = sequence_output[:, 0, :]
        x = self.dropout(cls_token)
        out = self.classifier(x)
        return out


class PooledModel(tf.keras.Model):
    def __init__(self, transformer):
        super(PooledModel, self).__init__()
        self.transformer = transformer
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.classifier = Dense(1, activation="sigmoid")

    def call(self, inputs):
        pooled_output = self.transformer(inputs)[1]
        x = self.dropout(pooled_output)
        out = self.classifier(x)
        return out


num_train_steps = math.ceil(len(X_train) / BATCH_SIZE) * EPOCHS

with strategy.scope():
    transformer = transformers.TFAutoModel.from_pretrained(MODEL_TYPE)
    model = PooledModel(transformer)
    optimizer = create_optimizer(LR, num_train_steps, 0.1 * num_train_steps)
    # model.compile(optimizer, loss=SscBinaryCrossEntropy(name="loss"), metrics=[SscAuc(name="auc")])
    model.compile(
        optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(name="loss"),
        metrics=[tf.keras.metrics.AUC(name="auc")],
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

train_history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    steps_per_epoch=4,
    validation_steps=1,
    callbacks=callbacks,
    epochs=2,
)

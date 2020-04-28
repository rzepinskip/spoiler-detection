import gzip
import hashlib
import json
import math
import os
import warnings
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import transformers
import wandb
from tqdm import tqdm
from wandb.keras import WandbCallback

from spoiler_detection import (
    SscAuc,
    SscWeightedBinaryCrossEntropy,
    WeightedBinaryCrossEntropy,
    create_optimizer,
)
from spoiler_detection.datasets import (
    GoodreadsSingleDataset,
    GoodreadsSscDataset,
    TvTropesMovieSingleDataset,
)
from spoiler_detection.models import PooledModel, SequenceModel, SscModel

MODELS = {
    "SequenceModel": SequenceModel,
    "PooledModel": PooledModel,
    "SscModel": SscModel,
}

DATASETS = {
    "GoodreadsSingleDataset": GoodreadsSingleDataset,
    "GoodreadsSscDataset": GoodreadsSscDataset,
    "TvTropesMovieSingleDataset": TvTropesMovieSingleDataset,
}


def get_callbacks(args):
    class LogLearningRate(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            lr = self.model.optimizer._decayed_lr("float32").numpy()
            wandb.log({"lr": lr})

    callbacks = list()
    if not args.dry_run:
        callbacks = []
        if not args.offline:
            wandb.init(
                project="spoiler_detection-keras", tags=[], config=args,
            )

            callbacks += [
                WandbCallback(monitor="val_auc", save_model=False),
                LogLearningRate(),
            ]
        callbacks += [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=False,
            )
        ]
        callbacks += [
            tf.keras.callbacks.ModelCheckpoint(
                f"{wandb.run.dir}/checkpoint.h5",
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
            )
        ]
    return callbacks


def main(args):
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    warnings.simplefilter("ignore")

    dataset = DATASETS[args.dataset_name](hparams=args)

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("Running on TPU")
    except ValueError:
        tpu = None
    strategy = tf.distribute.get_strategy()
    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)

    train_dataset_raw, y_train = dataset.get_dataset("train")
    val_dataset_raw, _ = dataset.get_dataset("val")

    train_dataset = (
        train_dataset_raw.shuffle(2048)
        .batch(args.batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    val_dataset = (
        val_dataset_raw.batch(args.batch_size)
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    num_train_steps = math.ceil(len(y_train) / args.batch_size) * args.epochs
    neg_count, pos_count = (len(y_train[y_train == 0]), len(y_train[y_train == 1]))
    pos_weight = neg_count / pos_count

    with strategy.scope():
        model = MODELS[args.model_name](hparams=args)
        optimizer = create_optimizer(
            args.learning_rate, num_train_steps, 0.1 * num_train_steps
        )
        model.compile(
            optimizer,
            # loss=tfa.losses.SigmoidFocalCrossEntropy(
            #     alpha=0.5,
            #     gamma=2.0,
            #     name="loss",
            #     reduction=tf.keras.losses.Reduction.AUTO,
            # ),
            # loss=tf.keras.losses.BinaryCrossentropy(name="loss"),
            # loss=WeightedBinaryCrossEntropy(pos_weight=pos_weight, name="loss"),
            loss=SscWeightedBinaryCrossEntropy(pos_weight=pos_weight, name="loss"),
            metrics=SscAuc()
            # metrics=[
            #     tf.keras.metrics.AUC(name="auc"),
            #     tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            #     tf.keras.metrics.BinaryAccuracy(name="acc"),
            #     # tfa.metrics.F1Score(num_classes=1, name="f1", threshold=0.5),
            #     tf.keras.metrics.TruePositives(name="tp"),
            #     tf.keras.metrics.FalsePositives(name="fp"),
            #     tf.keras.metrics.TrueNegatives(name="tn"),
            #     tf.keras.metrics.FalseNegatives(name="fn"),
            # ],
        )
        if tpu is None:
            model.run_eagerly = True

    callbacks = get_callbacks(args)
    if args.dry_run:
        train_history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            steps_per_epoch=4,
            validation_steps=2,
            callbacks=callbacks,
            epochs=2,
        )
    else:
        train_history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            callbacks=callbacks,
            epochs=args.epochs,
        )


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    parser.add_argument("--model_name", type=str, default="PooledModel")
    parser.add_argument(
        "--dataset_name", type=str, default="GoodreadsSingleEncodedGenreDataset"
    )
    parser.add_argument("--offline", type=int, choices={0, 1}, default=0)
    parser.add_argument("--dry_run", type=int, choices={0, 1}, default=0)
    parser.add_argument("--seed", type=int, default=44)

    parser.add_argument("--model_type", default="bert-base-uncased", type=str)
    parser.add_argument("--learning_rate", default=2e-6, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--epochs", default=4, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--max_sentences", default=5, type=int)
    args = parser.parse_args()

    main(args)

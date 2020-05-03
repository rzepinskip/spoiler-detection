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

from spoiler_detection import WeightedBinaryCrossEntropy, create_optimizer
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

    test_dataset_raw, _ = dataset.get_dataset("test")

    test_dataset = test_dataset_raw.batch(args.batch_size).prefetch(
        tf.data.experimental.AUTOTUNE
    )

    with strategy.scope():
        model = MODELS[args.model_name](hparams=args)
        optimizer = create_optimizer(args.learning_rate, 0, 0)
        model.compile(
            optimizer,
            loss=WeightedBinaryCrossEntropy(name="loss"),
            metrics=[
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
                tf.keras.metrics.BinaryAccuracy(name="acc"),
                tfa.metrics.F1Score(num_classes=1, name="f1", threshold=0.5),
                tf.keras.metrics.TruePositives(name="tp"),
                tf.keras.metrics.FalsePositives(name="fp"),
                tf.keras.metrics.TrueNegatives(name="tn"),
                tf.keras.metrics.FalseNegatives(name="fn"),
            ],
        )
    model.load_weights(args.checkpoint, by_name=True)
    test_history = model.evaluate(test_dataset, steps=2)
    print(dict(zip(model.metrics_names, test_history)))


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--model_name", type=str, default="PooledModel")
    parser.add_argument(
        "--dataset_name", type=str, default="TvTropesMovieSingleDataset"
    )
    parser.add_argument("--seed", type=int, default=44)

    parser.add_argument("--model_type", default="bert-base-uncased", type=str)
    parser.add_argument("--learning_rate", default=2e-6, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--max_sentences", default=5, type=int)
    args = parser.parse_args()

    main(args)

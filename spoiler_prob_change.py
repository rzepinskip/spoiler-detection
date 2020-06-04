import gzip
import json
import logging
import pickle
import warnings
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import tensorflow as tf
import transformers
from tqdm import tqdm

from spoiler_detection import WeightedBinaryCrossEntropy, create_optimizer
from spoiler_detection.datasets.utils import (
    encode,
    enforce_max_sent_per_example,
    get_model_group,
)
from spoiler_detection.models import PooledModel, SequenceModel, SscModel

MODELS = {
    "SequenceModel": SequenceModel,
    "PooledModel": PooledModel,
    "SscModel": SscModel,
}


DATA_SOURCES = {
    "train": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/tvtropes_books-train.json.gz",
    "val": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/tvtropes_books-val.json.gz",
    "test": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/tvtropes_books-test.json.gz",
}

LABELS_COUNTS = {0: 442475, 1: 89972}
BUCKET = "gs://spoiler-detection/tvtropes"


class TvTropesBookSingleDataset:
    def __init__(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "google/electra-base-discriminator", use_fast=True
        )
        self.max_length = 96

    def get_dataset(self, dataset_type):
        def all_words_spoiler(sentence, spoiler_indexes):
            if len(spoiler_indexes) > 1:
                return False
            spoiler_range = spoiler_indexes[0][1] - spoiler_indexes[0][0]
            return spoiler_range > 0.9 * len(sentence)

        X = list()
        y = list()
        ids = list()
        is_original = list()
        i = 0
        with gzip.open(transformers.cached_path(DATA_SOURCES[dataset_type])) as file:
            for line in file:
                tropes_json = json.loads(line)
                sentences, labels = list(), list()
                for spoiler, sentence, spoiler_indexes in tropes_json["sentences"]:
                    if (
                        spoiler
                        and not all_words_spoiler(sentence, spoiler_indexes)
                        and len(spoiler_indexes) == 1
                    ):
                        label = 1.0 if spoiler == True else 0.0
                        X.append(sentence)
                        y.append(label)
                        ids.append(i)
                        is_original.append(True)
                        for start, end in spoiler_indexes:
                            cut_out_sentence = ""
                            if start == 0:
                                cut_out_sentence = sentence[end + 1 :]
                            else:
                                cut_out_sentence = (
                                    sentence[:start] + " " + sentence[end + 1 :]
                                )
                            X.append(cut_out_sentence)
                            y.append(label)
                            ids.append(i)
                            is_original.append(False)
                        i += 1
        X = {
            "input_ids": encode(X, self.tokenizer, self.max_length),
            "id": ids,
            "is_original": is_original,
        }
        y = np.array(y)
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        labels_count = {0: sum(y == 0), 1: sum(y == 1)}
        return dataset, labels_count


parser = ArgumentParser(add_help=False)

parser.add_argument("--checkpoint", type=str, default="TvTropesBookSingleDataset.h5")
parser.add_argument("--model_name", type=str, default="SequenceModel")
parser.add_argument("--seed", type=int, default=44)

parser.add_argument(
    "--model_type", default="google/electra-base-discriminator", type=str
)
parser.add_argument("--learning_rate", default=2e-6, type=float)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--max_length", default=128, type=int)
parser.add_argument("--use_genres", type=int, choices={0, 1}, default=0)
args = parser.parse_args([])


tf.random.set_seed(args.seed)
np.random.seed(args.seed)
warnings.simplefilter("ignore")

dataset = TvTropesBookSingleDataset()
train_dataset_raw, labels_count = dataset.get_dataset("val")

train_dataset = (
    train_dataset_raw.batch(args.batch_size)
    .take(10)
    .prefetch(tf.data.experimental.AUTOTUNE)
)
strategy = tf.distribute.get_strategy()

with strategy.scope():
    model = MODELS[args.model_name](hparams=args, output_bias=0)

    optimizer = create_optimizer(args.learning_rate, 0, 0)
    loss = WeightedBinaryCrossEntropy(pos_weight=1, name="loss")

    model.compile(
        optimizer, loss=loss, metrics=[tf.keras.metrics.AUC(name="auc"),],
    )


model.predict({"input_ids": encode(["Test."], dataset.tokenizer)})

model.load_weights(args.checkpoint, by_name=True)

predictions = model.predict(train_dataset)

mod_change = []
for index, element in train_dataset.enumerate():
    predictions_index = 2 * index
    truth = predictions[predictions_index]
    modified = predictions[predictions_index + 1]
    mod_change += [modified < truth]

print(f"{sum(mod_change)} of {len(mod_change)}")

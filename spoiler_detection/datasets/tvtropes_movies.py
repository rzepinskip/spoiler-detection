import csv

import numpy as np
import tensorflow as tf
import transformers

from .utils import encode, enforce_max_sent_per_example

DATA_SOURCES = {
    "train": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/tvtropes_movie-train.balanced.csv",
    "val": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/tvtropes_movie-dev1.balanced.csv",
    "test": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/tvtropes_movie-test.balanced.csv",
}


class TvTropesMovieSingleDataset:
    def __init__(self, hparams):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            hparams.model_type, use_fast=True
        )
        self.max_length = hparams.max_length

    def get_dataset(self, dataset_type):
        X = list()
        y = list()
        with open(transformers.cached_path(DATA_SOURCES[dataset_type])) as file:
            reader = csv.reader(file)
            next(reader)  # skip header
            for sentence, spoiler, verb, page, trope in reader:
                label = 1.0 if spoiler == "True" else 0.0
                X.append(sentence)
                y.append(label)

        X = {"input_ids": encode(X, self.tokenizer, self.max_length)}
        y = np.array(y)
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        labels_count = {0: sum(y == 0), 1: sum(y == 1)}
        return dataset, labels_count

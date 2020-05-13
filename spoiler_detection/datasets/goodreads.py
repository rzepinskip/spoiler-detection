import gzip
import json
import logging

import numpy as np
import tensorflow as tf
import transformers

from ..feature_encoders import encode_as_distribution, encode_as_string

LABELS_COUNTS = {0: 2110317, 1: 455921}
BUCKET = "gs://spoiler-detection/genres"


def get_model_group(model_type):
    if model_type.startswith("bert"):
        name_split = model_type.split("-")
        return f"{name_split[0]}_{name_split[2]}"

    model_group = model_type.split("-")[0]

    dash_sep = model_type.find("/")
    if dash_sep:
        model_group = model_group[dash_sep + 1 :]

    return model_group


class GoodreadsSingleGenreAppendedDataset:
    def __init__(self, hparams):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            hparams.model_type, use_fast=True
        )
        self.max_length = hparams.max_length
        self.model_group = get_model_group(hparams.model_type)
        self.prefix = hparams.prefix

    def get_dataset(self, dataset_type):
        file_path = f"{BUCKET}/{self.prefix}-{self.get_file_name(dataset_type)}"
        print(f"Training on: {file_path}")

        def read_tfrecord(serialized_example):
            feature_description = {
                "input_ids": tf.io.FixedLenFeature([], tf.string),
                "genres": tf.io.FixedLenFeature([], tf.string),
                "label": tf.io.FixedLenFeature([], tf.string),
            }

            example = tf.io.parse_single_example(
                serialized_example, feature_description
            )
            input_ids = tf.ensure_shape(
                tf.io.parse_tensor(example["input_ids"], tf.int32), [self.max_length]
            )
            genres = tf.ensure_shape(
                tf.io.parse_tensor(example["genres"], tf.float32), [10]
            )
            label = tf.ensure_shape(
                tf.io.parse_tensor(example["label"], tf.float32), []
            )

            return {"input_ids": input_ids, "genres": genres}, label

        dataset = tf.data.TFRecordDataset(file_path, compression_type="GZIP").map(
            read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        return dataset, LABELS_COUNTS

    def get_file_name(self, dataset_type):
        return f"{GoodreadsSingleGenreAppendedDataset.__name__}-{self.model_group}-{self.max_length}-{dataset_type}.tf.gz"

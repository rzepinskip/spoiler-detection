import gzip
import json
import logging

import numpy as np
import tensorflow as tf
import transformers

from .utils import encode, enforce_max_sent_per_example, get_model_group

DATA_SOURCES = {
    "train": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/tvtropes_books-train.json.gz",
    "val": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/tvtropes_books-val.json.gz",
    "test": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/tvtropes_books-test.json.gz",
}

LABELS_COUNTS = {0: 442475, 1: 89972}
BUCKET = "gs://spoiler-detection/tvtropes"


class TvTropesBookSingleDataset:
    def __init__(self, hparams):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            hparams.model_type, use_fast=True
        )
        self.max_length = hparams.max_length

    def get_dataset(self, dataset_type):
        X = list()
        y = list()
        with gzip.open(transformers.cached_path(DATA_SOURCES[dataset_type])) as file:
            for line in file:
                tropes_json = json.loads(line)
                sentences, labels = list(), list()
                for spoiler, sentence, _ in tropes_json["sentences"]:
                    label = 1.0 if spoiler == True else 0.0
                    X.append(sentence)
                    y.append(label)

        X = {"input_ids": encode(X, self.tokenizer, self.max_length)}
        y = np.array(y)
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        labels_count = {0: sum(y == 0), 1: sum(y == 1)}
        return dataset, labels_count


class TvTropesBookSscDataset:
    def __init__(self, hparams):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            hparams.model_type, use_fast=True
        )
        self.max_length = hparams.max_length
        self.max_sentences = hparams.max_sentences
        self.model_group = get_model_group(hparams.model_type)
        self.sep_id = self.tokenizer.convert_tokens_to_ids(["[SEP]"])[0]

        if self.model_group not in ("bert_cased", "bert_uncased", "albert", "electra"):
            raise ValueError(
                "This model works only for BERT-like input models with SEP and CLS tokens"
            )

    def get_file_name(self, dataset_type):
        return f"{TvTropesBookSscDataset.__name__}-{self.model_group}-{self.max_length}-{self.max_sentences}-{dataset_type}.tf.gz"

    def process(self, line):
        tropes_json = json.loads(line.numpy())
        raw_sentences, raw_labels = list(), list()
        for label, sentence, _ in tropes_json["sentences"]:
            raw_labels.append(float(label))
            raw_sentences.append(sentence)

        sentences, labels = list(), list()
        for (sentences_loop, labels_loop) in enforce_max_sent_per_example(
            raw_sentences, labels=raw_labels, max_sentences=self.max_sentences,
        ):
            sentences.append("[SEP]".join(sentences_loop))
            labels.append(labels_loop)

        input_ids = encode(sentences, self.tokenizer, self.max_length,)

        if any([x[self.max_length - 1] != 0 for x in input_ids]):
            input_ids = np.array(input_ids)
            sentences_sums = np.sum(input_ids == self.sep_id, axis=1,)
            labels_sums = [len(x) for x in labels]
            for i in range(len(labels)):
                s = sentences_sums[i]
                l = labels_sums[i]
                if s != l:
                    logging.debug(f"[#{i}] Truncating. Original:\n {sentences[i]}")
                    labels[i] = labels[i][:s]
        indices = tf.where(input_ids == self.sep_id)
        updates = [item for sublist in labels for item in sublist]
        labels_scattered = tf.tensor_scatter_nd_update(
            tf.constant(-1.0, shape=tf.shape(input_ids)), indices, updates
        )

        return input_ids, labels_scattered

    def write_dataset(self, dataset_type):
        dataset = (
            tf.data.TextLineDataset(
                transformers.cached_path(DATA_SOURCES[dataset_type]),
                compression_type="GZIP",
            )
            .map(
                lambda x: tf.py_function(self.process, [x], [tf.int32, tf.float32]),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .interleave(
                lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)),
                cycle_length=1,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        )

        def serialize_example(input_ids, label):
            def _bytes_feature(value):
                return tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[value.numpy()])
                )

            feature = {
                "input_ids": _bytes_feature(tf.io.serialize_tensor(input_ids)),
                "label": _bytes_feature(tf.io.serialize_tensor(label)),
            }

            example_proto = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )
            return example_proto.SerializeToString()

        dataset = dataset.map(
            lambda x, y: tf.py_function(serialize_example, [x, y], [tf.string])[0],
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        file_path = f"./{self.get_file_name(dataset_type)}"
        writer = tf.data.experimental.TFRecordWriter(file_path, compression_type="GZIP")
        writer.write(dataset)

    def get_dataset(self, dataset_type):
        file_path = f"{BUCKET}/{self.get_file_name(dataset_type)}"
        # self.write_dataset(dataset_type)
        # file_path = f"./{self.get_file_name(dataset_type)}"

        def read_tfrecord(serialized_example):
            feature_description = {
                "input_ids": tf.io.FixedLenFeature([], tf.string),
                "label": tf.io.FixedLenFeature([], tf.string),
            }

            example = tf.io.parse_single_example(
                serialized_example, feature_description
            )
            input_ids = tf.ensure_shape(
                tf.io.parse_tensor(example["input_ids"], tf.int32), [self.max_length]
            )
            label = tf.ensure_shape(
                tf.io.parse_tensor(example["label"], tf.float32), [self.max_length]
            )

            return {"input_ids": input_ids}, label

        dataset = tf.data.TFRecordDataset(file_path, compression_type="GZIP").map(
            read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        return dataset, LABELS_COUNTS

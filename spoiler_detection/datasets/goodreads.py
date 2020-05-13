import gzip
import json
import logging

import numpy as np
import tensorflow as tf
import transformers

from ..feature_encoders import encode_as_distribution, encode_as_string

DATA_SOURCES = {
    "train": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-train.json.gz",
    "val": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-val.json.gz",
    "test": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-test.json.gz",
}
LABELS_COUNTS = {0: 2110317, 1: 455921}
BUCKET = "gs://spoiler-detection/goodreads"


def encode(texts, tokenizer, max_length=512):
    input_ids = tokenizer.batch_encode_plus(
        texts,
        return_attention_masks=False,
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=max_length,
    )["input_ids"]

    return np.array(input_ids)


def get_model_group(model_type):
    if model_type.startswith("bert"):
        name_split = model_type.split("-")
        return f"{name_split[0]}_{name_split[2]}"

    model_group = model_type.split("-")[0]

    dash_sep = model_type.find("/")
    if dash_sep:
        model_group = model_group[dash_sep + 1 :]

    return model_group


class GoodreadsSingleDataset:
    def __init__(self, hparams):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            hparams.model_type, use_fast=True
        )
        self.max_length = hparams.max_length
        self.model_group = get_model_group(hparams.model_type)

    def get_file_name(self, dataset_type):
        return f"{GoodreadsSingleDataset.__name__}-{self.model_group}-{self.max_length}-{dataset_type}.tf.gz"

    def process(self, line):
        review_json = json.loads(line.numpy())
        sentences, labels = list(), list()
        genres = review_json["genres"]
        encoded_genres = encode_as_distribution(genres)
        for label, sentence in review_json["review_sentences"]:
            sentences.append(sentence)
            labels.append(float(label))
        input_ids = encode(sentences, self.tokenizer, self.max_length,)

        return input_ids, [encoded_genres for _ in labels], labels

    def write_dataset(self, dataset_type):
        dataset = (
            tf.data.TextLineDataset(
                transformers.cached_path(DATA_SOURCES[dataset_type]),
                compression_type="GZIP",
            )
            .map(
                lambda x: tf.py_function(
                    self.process, [x], [tf.int32, tf.float32, tf.float32]
                ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .interleave(
                lambda x, y, z: tf.data.Dataset.from_tensor_slices((x, y, z)),
                cycle_length=1,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        )

        def serialize_example(input_ids, genres, label):
            def _bytes_feature(value):
                return tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[value.numpy()])
                )

            feature = {
                "input_ids": _bytes_feature(tf.io.serialize_tensor(input_ids)),
                "genres": _bytes_feature(tf.io.serialize_tensor(genres)),
                "label": _bytes_feature(tf.io.serialize_tensor(label)),
            }

            example_proto = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )
            return example_proto.SerializeToString()

        dataset = dataset.map(
            lambda x, y, z: tf.py_function(serialize_example, [x, y, z], [tf.string])[
                0
            ],
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


class GoodreadsSingleGenreAppendedDataset(GoodreadsSingleDataset):
    def __init__(self, hparams):
        super().__init__(hparams)

    def get_file_name(self, dataset_type):
        return f"{GoodreadsSingleGenreAppendedDataset.__name__}-{self.model_group}-{self.max_length}-{dataset_type}.tf.gz"

    def process(self, line):
        review_json = json.loads(line.numpy())
        genres = review_json["genres"]
        sentences, labels = list(), list()
        encoded_genres = encode_as_distribution(genres)
        for label, sentence in review_json["review_sentences"]:
            sentences.append(
                f"{sentence}[SEP]{encode_as_string(genres)}"
            )  # TODO not always SEP!
            labels.append(float(label))
        input_ids = encode(sentences, self.tokenizer, self.max_length,)
        return input_ids, [encoded_genres for _ in labels], labels


def enforce_max_sent_per_example(sentences, labels, max_sentences=1):
    assert len(sentences) == len(labels)

    chunks = (
        len(sentences) // max_sentences
        if len(sentences) % max_sentences == 0
        else len(sentences) // max_sentences + 1
    )
    return zip(np.array_split(sentences, chunks), np.array_split(labels, chunks))


class GoodreadsSscDataset:
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
        return f"{GoodreadsSscDataset.__name__}-{self.model_group}-{self.max_length}-{self.max_sentences}-{dataset_type}.tf.gz"

    def process(self, line):
        review_json = json.loads(line.numpy())
        raw_sentences, raw_labels = list(), list()
        for label, sentence in review_json["review_sentences"]:
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


class GoodreadsSscGenreAppendedDataset(GoodreadsSscDataset):
    def __init__(self, hparams):
        super().__init__(hparams)

    def get_file_name(self, dataset_type):
        return f"{GoodreadsSscGenreAppendedDataset.__name__}-{self.model_group}-{self.max_length}-{self.max_sentences}-{dataset_type}.tf.gz"

    def process(self, line):
        review_json = json.loads(line.numpy())
        raw_sentences, raw_labels = list(), list()
        for label, sentence in review_json["review_sentences"]:
            raw_labels.append(float(label))
            raw_sentences.append(sentence)

        genres = review_json["genres"]
        genres_encoded = encode_as_string(genres)

        sentences, labels = list(), list()
        for (sentences_loop, labels_loop) in enforce_max_sent_per_example(
            raw_sentences, labels=raw_labels, max_sentences=self.max_sentences,
        ):
            sentences_sequence = f"[CLS]{'[SEP]'.join(sentences_loop)}[SEP]{genres_encoded}"  # omit final [SEP] on purpose
            sentences.append(sentences_sequence)
            labels.append(labels_loop)

        def special_encode(texts, tokenizer, max_length=512):
            input_ids = tokenizer.batch_encode_plus(
                texts,
                return_attention_masks=False,
                return_token_type_ids=False,
                pad_to_max_length=True,
                max_length=max_length,
                add_special_tokens=False,
            )["input_ids"]

            return np.array(input_ids)

        input_ids = special_encode(sentences, self.tokenizer, self.max_length,)

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

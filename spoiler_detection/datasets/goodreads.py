import gzip
import json
import logging

import numpy as np
import tensorflow as tf
import transformers

from ..feature_encoders import encode_as_distribution, encode_as_string

DATA_SOURCES = {
    "train": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-timings-train.json.gz",
    "val": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-timings-val.json.gz",
    "test": None,
}


def encode(texts, tokenizer, max_length=512):
    input_ids = tokenizer.batch_encode_plus(
        texts,
        return_attention_masks=False,
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=max_length,
    )["input_ids"]

    return np.array(input_ids)


class GoodreadsSingleDataset:
    def __init__(self, hparams):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            hparams.model_type, use_fast=True
        )
        self.max_length = hparams.max_length

    def get_dataset(self, dataset_type):
        all_sentences, all_genres, all_labels = list(), list(), list()
        with gzip.open(transformers.cached_path(DATA_SOURCES[dataset_type])) as file:
            for line in file:
                review_json = json.loads(line)
                genres = review_json["genres"]
                sentences, labels = list(), list()
                for label, sentence in review_json["review_sentences"]:
                    all_sentences.append(sentence)
                    all_genres.append(encode_as_distribution(genres))
                    all_labels.append(float(label))

        X = {
            "sentence": encode(all_sentences, self.tokenizer, self.max_length),
            "genres": all_genres,
        }
        y = np.array(all_labels)
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        return dataset, y


class GoodreadsSingleGenreAppendedDataset:
    def __init__(self, hparams):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            hparams.model_type, use_fast=True
        )
        self.max_length = hparams.max_length

    def get_dataset(self, dataset_type):
        all_sentences, all_genres, all_labels = list(), list(), list()
        with gzip.open(transformers.cached_path(DATA_SOURCES[dataset_type])) as file:
            for line in file:
                review_json = json.loads(line)
                genres = review_json["genres"]
                sentences, labels = list(), list()
                for label, sentence in review_json["review_sentences"]:
                    all_sentences.append(f"{sentence}[SEP]{encode_as_string(genres)}")
                    all_genres.append(encode_as_distribution(genres))
                    all_labels.append(float(label))

        X = {
            "sentence": encode(all_sentences, self.tokenizer, self.max_length),
            "genres": all_genres,
        }
        y = np.array(all_labels)
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        return dataset, y


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
        self.sep_id = self.tokenizer.convert_tokens_to_ids(["[SEP]"])[0]

    def get_dataset(self, dataset_type):
        X, y = list(), list()
        y_true = list()
        with gzip.open(transformers.cached_path(DATA_SOURCES[dataset_type])) as file:
            for line in file:
                review_json = json.loads(line)
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
                    y_true.extend(labels_loop)

                output = encode(sentences, self.tokenizer, self.max_length,)

                if any([x[self.max_length - 1] != 0 for x in output]):
                    input_ids = np.array(output)
                    sentences_sums = np.sum(input_ids == self.sep_id, axis=1,)
                    labels_sums = [len(x) for x in labels]
                    for i in range(len(labels)):
                        s = sentences_sums[i]
                        l = labels_sums[i]
                        if s != l:
                            logging.debug(
                                f"[#{i}] Truncating. Original:\n {sentences[i]}"
                            )
                            labels[i] = labels[i][:s]
                indices = tf.where(output == self.sep_id)
                updates = [item for sublist in labels for item in sublist]
                labels_scattered = tf.tensor_scatter_nd_update(
                    tf.constant(-1.0, shape=tf.shape(output)), indices, updates
                )
                X.extend(output)
                y.extend(tf.expand_dims(labels_scattered, -1))

        dataset = tf.data.Dataset.from_tensor_slices((np.array(X), y))
        return dataset, np.array(y_true)

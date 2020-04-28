import gzip
import json
import logging

import numpy as np
import tensorflow as tf
import transformers

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
        X = list()
        y = list()
        with gzip.open(transformers.cached_path(DATA_SOURCES[dataset_type])) as file:
            for line in file:
                review_json = json.loads(line)
                genres = review_json["genres"]
                sentences, labels = list(), list()
                for label, sentence in review_json["review_sentences"]:
                    X.append(sentence)
                    y.append(float(label))

        X = encode(X, self.tokenizer, self.max_length)
        y = np.array(y)
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
            hparams.model_type, use_fast=False
        )
        self.max_length = hparams.max_length
        self.max_sentences = hparams.max_sentences

    def get_dataset(self, dataset_type):
        X = list()
        X_weights = list()
        y = list()
        y_true = list()
        with gzip.open(transformers.cached_path(DATA_SOURCES[dataset_type])) as file:
            for line in file:
                review_json = json.loads(line)
                genres = review_json["genres"]
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
                    sentences_sums = np.sum(
                        input_ids == self.tokenizer._convert_token_to_id("[SEP]"),
                        axis=1,
                    )
                    labels_sums = [len(x) for x in labels]
                    for i in range(len(labels)):
                        s = sentences_sums[i]
                        l = labels_sums[i]
                        if s != l:
                            logging.debug(
                                f"[#{i}] Truncating. Original:\n {sentences[i]}"
                            )
                            labels[i] = labels[i][:s]
                sample_weight = 1.0 * (output == 102)
                indices = tf.where(sample_weight != 0)
                updates = [item for sublist in labels for item in sublist]
                labels_scattered = tf.tensor_scatter_nd_update(
                    -1.0 * tf.ones_like(sample_weight), indices, updates
                )
                X.extend(output)
                X_weights.extend(sample_weight)
                y.extend(tf.expand_dims(labels_scattered, -1))

        dataset = tf.data.Dataset.from_tensor_slices(
            (np.array(X), y, np.array(X_weights))
        )
        return dataset, np.array(y_true)

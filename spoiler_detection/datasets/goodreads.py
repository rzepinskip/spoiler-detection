import gzip
import json

import numpy as np
import tensorflow as tf
import transformers


def enforce_max_sent_per_example(sentences, labels, max_sentences=1):
    assert len(sentences) == len(labels)

    chunks = (
        len(sentences) // max_sentences
        if len(sentences) % max_sentences == 0
        else len(sentences) // max_sentences + 1
    )
    return zip(np.array_split(sentences, chunks), np.array_split(labels, chunks))


def encode(texts, tokenizer, max_length=512):
    input_ids = tokenizer.batch_encode_plus(
        texts,
        return_attention_masks=False,
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=max_length,
    )["input_ids"]

    return np.array(input_ids)


def get_goodreads_single(path, tokenizer, max_length):
    X = list()
    y = list()
    with gzip.open(transformers.cached_path(path)) as file:
        for line in file:
            review_json = json.loads(line)
            genres = review_json["genres"]
            sentences, labels = list(), list()
            for label, sentence in review_json["review_sentences"]:
                X.append(sentence)
                y.append(label)

    X = encode(X, tokenizer, max_length)
    y = np.array(y)
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    return dataset, y


def get_goodreads_ssc(path, tokenizer, max_length):
    X = list()
    y = list()
    with gzip.open(transformers.cached_path(path)) as file:
        for line in file:
            review_json = json.loads(line)
            genres = review_json["genres"]
            sentences, labels = list(), list()
            for label, sentence in review_json["review_sentences"]:
                labels.append(label)
                sentences.append(sentence)

            for (sentences_loop, labels_loop) in enforce_max_sent_per_example(
                sentences, labels=labels, max_sentences=5,
            ):
                X.append("[SEP]".join(sentences_loop))
                y.append(labels_loop)

    X = np.array(encode(X, tokenizer, max_length))
    y = tf.keras.preprocessing.sequence.pad_sequences(y, padding="post", value=-1)
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    return dataset, y

import gzip
import json

import numpy as np
import tensorflow as tf
from transformers import cached_path


def enforce_max_sent_per_example(sentences, labels=None, max_sentences=0):
    if labels is not None:
        assert len(sentences) == len(labels)

    chunks = (
        len(sentences) // max_sentences
        if len(sentences) % max_sentences == 0
        else len(sentences) // max_sentences + 1
    )
    return zip(np.array_split(sentences, chunks), np.array_split(labels, chunks))


def get_goodreads(path):
    X = list()
    y = list()
    with gzip.open(cached_path(path)) as file:
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

    X = np.array(X)
    y = tf.keras.preprocessing.sequence.pad_sequences(y, padding="post", value=-1)
    return X, y

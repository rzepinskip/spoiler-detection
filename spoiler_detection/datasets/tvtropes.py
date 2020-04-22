import csv

import numpy as np
import tensorflow as tf
import transformers


def encode(texts, tokenizer, max_length=512):
    input_ids = tokenizer.batch_encode_plus(
        texts,
        return_attention_masks=False,
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=max_length,
    )["input_ids"]

    return np.array(input_ids)


def get_tvtropes_movie_single(path, tokenizer, max_length):
    X = list()
    y = list()
    with open(transformers.cached_path(path)) as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        for sentence, spoiler, verb, page, trope in reader:
            label = 1 if spoiler == "True" else 0
            X.append(sentence)
            y.append(label)

    X = encode(X, tokenizer, max_length)
    y = np.array(y)
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    return dataset, y

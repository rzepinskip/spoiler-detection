import pickle
import warnings
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import tensorflow as tf
import transformers
from tqdm import tqdm

from spoiler_detection import WeightedBinaryCrossEntropy, create_optimizer
from spoiler_detection.datasets import (
    GoodreadsSingleDataset,
    GoodreadsSingleGenreAppendedDataset,
    GoodreadsSscDataset,
    GoodreadsSscGenreAppendedDataset,
    TvTropesBookSingleDataset,
    TvTropesBookSscDataset,
    TvTropesMovieSingleDataset,
)
from spoiler_detection.models import PooledModel, SequenceModel, SscModel

MODELS = {
    "SequenceModel": SequenceModel,
    "PooledModel": PooledModel,
    "SscModel": SscModel,
}

DATASETS = {
    "GoodreadsSingleDataset": GoodreadsSingleDataset,
    "GoodreadsSingleGenreAppendedDataset": GoodreadsSingleGenreAppendedDataset,
    "GoodreadsSscDataset": GoodreadsSscDataset,
    "GoodreadsSscGenreAppendedDataset": GoodreadsSscGenreAppendedDataset,
    "TvTropesMovieSingleDataset": TvTropesMovieSingleDataset,
    "TvTropesBookSingleDataset": TvTropesBookSingleDataset,
    "TvTropesBookSscDataset": TvTropesBookSscDataset,
}


parser = ArgumentParser(add_help=False)

parser.add_argument(
    "--checkpoint", type=str, default="GoodreadsSingleGenreAppendedDataset.h5"
)
parser.add_argument("--model_name", type=str, default="SequenceModel")
parser.add_argument(
    "--dataset_name", type=str, default="GoodreadsSingleGenreAppendedDataset"
)
parser.add_argument("--seed", type=int, default=44)

parser.add_argument(
    "--model_type", default="google/electra-base-discriminator", type=str
)
parser.add_argument("--learning_rate", default=2e-6, type=float)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--max_length", default=128, type=int)
parser.add_argument("--use_genres", type=int, choices={0, 1}, default=0)
args = parser.parse_args([])


tf.random.set_seed(args.seed)
np.random.seed(args.seed)
warnings.simplefilter("ignore")

dataset = DATASETS[args.dataset_name](hparams=args)
train_dataset_raw, labels_count = dataset.get_dataset("train")

train_dataset = (
    train_dataset_raw.prefetch(tf.data.experimental.AUTOTUNE)
    .shuffle(8096, seed=args.seed, reshuffle_each_iteration=True)
    .batch(args.batch_size)
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


def encode(texts, tokenizer, max_length=512):
    input_ids = tokenizer.batch_encode_plus(
        texts,
        return_attention_masks=False,
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=max_length,
    )["input_ids"]

    return {"input_ids": np.array(input_ids)}


model.call(encode(["Test string 1."], tokenizer=dataset.tokenizer))

model.load_weights(args.checkpoint, by_name=True)
results = []
batches = 1000
for index, element in tqdm(train_dataset.take(batches).enumerate(), total=batches):
    input_data, label = element
    attn = model.get_attention(input_data)
    cls_attn = tf.reduce_mean(attn[-1][:, 0, :], 1)

    label = label.numpy()
    tokens = input_data["input_ids"].numpy()
    cls_attn = cls_attn.numpy()
    for i in range(args.batch_size):
        results += [(label[i], tokens[i], cls_attn[i])]

with open("attentions.pickle", "wb") as handle:
    pickle.dump(results, handle)

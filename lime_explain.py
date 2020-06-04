import warnings
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
import transformers

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

parser.add_argument("--checkpoint", type=str, default="TvTropesBookSingleDataset.h5")
parser.add_argument("--model_name", type=str, default="SequenceModel")
parser.add_argument("--dataset_name", type=str, default="TvTropesBookSingleDataset")
parser.add_argument("--seed", type=int, default=44)

parser.add_argument(
    "--model_type", default="google/electra-base-discriminator", type=str
)
parser.add_argument("--learning_rate", default=2e-6, type=float)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--max_length", default=128, type=int)
parser.add_argument("--use_genres", type=int, choices={0, 1}, default=0)
args = parser.parse_args([])


tf.random.set_seed(args.seed)
np.random.seed(args.seed)
warnings.simplefilter("ignore")

dataset = DATASETS[args.dataset_name](hparams=args)

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


def get_proba(texts):
    encoded = encode(texts, dataset.tokenizer, max_length=args.max_length)
    res = model.predict(encoded)
    probs = [[1 - x[0], x[0]] for x in res]
    return probs


get_proba(["Test string 1."])

model.load_weights(args.checkpoint, by_name=True)


res = get_proba(["Test string 1.", "Test string 2", "Test string 3"])
res

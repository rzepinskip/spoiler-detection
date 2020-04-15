import gzip
import itertools
import json
import logging
from functools import partial

import torch
from torch import tensor
from torch.utils.data import DataLoader
from transformers import cached_path

from spoiler_detection.datasets.base_dataset import BaseDataset, ListDataset
from spoiler_detection.datasets.datasets_maps import get_goodreads_map
from spoiler_detection.datasets.utils import enforce_max_sent_per_example, pad_sequence
from spoiler_detection.feature_encoders import encode_genre


class GoodreadsSingleSentenceDataset(BaseDataset):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def prepare_sample(self, tokenizer, samples):
        sentences = [x["sentence"] for x in samples]
        labels = [x["label"] for x in samples]
        genres = [encode_genre(x["genre"]) for x in samples]

        output = tokenizer.batch_encode_plus(
            sentences, max_length=self.hparams.max_length, pad_to_max_length=True
        )
        return (
            tensor(output["input_ids"]),
            tensor(output["attention_mask"]),
            tensor(output["token_type_ids"]),
            tensor(genres),
            tensor(labels),
        )

    def get_dataloader(self, dataset_type, tokenizer, batch_size):
        data = []
        with gzip.open(cached_path(get_goodreads_map()[dataset_type])) as file:
            for line in file:
                review_json = json.loads(line)
                genres = review_json["genres"]
                for label, sentence in review_json["review_sentences"]:
                    data.append({"label": label, "sentence": sentence, "genre": genres})

        dataset = ListDataset(data)
        return DataLoader(
            dataset,
            num_workers=self.hparams.num_workers,
            collate_fn=partial(self.prepare_sample, tokenizer),
            batch_size=batch_size,
            shuffle=True,
        )


class GoodreadsMultiSentenceDataset(BaseDataset):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def prepare_sample(self, tokenizer, samples):
        genres = [encode_genre(x["genre"]) for x in samples]
        encoded_sentences = [
            tokenizer.batch_encode_plus(
                s["sentences"],
                max_length=self.hparams.max_length,
                pad_to_max_length=True,
            )
            for s in samples
        ]

        return (
            pad_sequence(
                [tensor(es["input_ids"]) for es in encoded_sentences], padding_value=0,
            ),
            pad_sequence(
                [tensor(es["attention_mask"]) for es in encoded_sentences],
                padding_value=False,
            ),
            pad_sequence(
                [tensor(es["token_type_ids"]) for es in encoded_sentences],
                padding_value=0,
            ),
            tensor(genres),
            pad_sequence([tensor(x["labels"]) for x in samples]),
        )

    def get_dataloader(self, dataset_type, tokenizer, batch_size):
        data = []
        with gzip.open(cached_path(get_goodreads_map()[dataset_type])) as file:
            for line in file:
                review_json = json.loads(line)
                genres = review_json["genres"]
                sentences, labels = list(), list()
                for label, sentence in review_json["review_sentences"]:
                    labels.append(label)
                    sentences.append(sentence)

                data.append({"labels": labels, "sentences": sentences, "genre": genres})

        dataset = ListDataset(data)
        return DataLoader(
            dataset,
            num_workers=self.hparams.num_workers,
            collate_fn=partial(self.prepare_sample, tokenizer),
            batch_size=batch_size,
            shuffle=True,
        )


class GoodreadsSscDataset(BaseDataset):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def prepare_sample(self, tokenizer, samples):
        sentences = ["[SEP]".join(x["sentences"]) for x in samples]

        output = tokenizer.batch_encode_plus(
            sentences, max_length=self.hparams.max_length, pad_to_max_length=True
        )
        input_ids = tensor(output["input_ids"])
        num_sentences = sum(sum(input_ids == tokenizer._convert_token_to_id("[SEP]")))
        num_labels = sum([len(x["labels"]) for x in samples])

        if num_sentences != num_labels:
            for i in range(len(samples)):
                s = len([x for x in output["input_ids"][i] if x == 3])
                l = len(samples[i]["labels"])
                if s != l:
                    logging.debug(
                        f"\t Sentence #{i} is too long. Truncating. Original:\n {sentences[i]}"
                    )
                    samples[i]["labels"] = samples[i]["labels"][:s]

        return (
            input_ids,
            tensor(output["attention_mask"]),
            tensor(output["token_type_ids"]),
            pad_sequence(
                [
                    tensor([x["genre"] for _ in range(len(x["sentences"]))])
                    for x in samples
                ],
                max_length=self.hparams.max_length,
            ),
            pad_sequence(
                [tensor(x["labels"]) for x in samples],
                max_length=self.hparams.max_length,
            ),
        )

    def get_dataloader(self, dataset_type, tokenizer, batch_size):
        data = []
        with gzip.open(cached_path(get_goodreads_map()[dataset_type])) as file:
            for line in file:
                review_json = json.loads(line)
                genres = review_json["genres"]
                sentences, labels = list(), list()
                for label, sentence in review_json["review_sentences"]:
                    labels.append(label)
                    sentences.append(sentence)

                for (sentences_loop, labels_loop) in enforce_max_sent_per_example(
                    sentences,
                    labels=labels,
                    max_sentences=self.hparams.max_sent_per_example,
                ):
                    data.append(
                        {
                            "labels": labels_loop,
                            "sentences": sentences_loop,
                            "genre": encode_genre(genres),
                        }
                    )

        dataset = ListDataset(data)
        return DataLoader(
            dataset,
            num_workers=self.hparams.num_workers,
            collate_fn=partial(self.prepare_sample, tokenizer),
            batch_size=batch_size,
            shuffle=True,
        )

    @classmethod
    def add_dataset_specific_args(cls, parent_parser):
        parser = BaseDataset.add_dataset_specific_args(parent_parser)
        parser.add_argument("--max_sent_per_example", type=int, default=5)
        return parser

import gzip
import itertools
import json
import logging

from torch import tensor
from torch.utils.data import DataLoader
from transformers import cached_path

from spoiler_detection.datasets.base_dataset import BaseDataset, ListDataset
from spoiler_detection.datasets.utils import pad_sequence
from spoiler_detection.feature_encoders import encode_genre

DATASET_MAP = {
    "train": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-train.json.gz",
    "val": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-val.json.gz",
    "test": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-test.json.gz",
    "dev": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads-lite.json.gz",
}


class GoodreadsSingleSentenceDataset(BaseDataset):
    def __init__(self, hparams):
        super().__init__()
        self._max_length = hparams.max_length

    def get_dataloader(self, dataset_type, tokenizer, batch_size):
        def prepare_sample(samples):
            sentences = [x["sentence"] for x in samples]
            labels = [x["label"] for x in samples]
            genres = [encode_genre(x["genre"]) for x in samples]

            output = tokenizer.batch_encode_plus(
                sentences, max_length=self._max_length, pad_to_max_length=True
            )
            return (
                tensor(output["input_ids"]),
                tensor(output["attention_mask"]),
                tensor(output["token_type_ids"]),
                tensor(genres),
                tensor(labels),
            )

        data = []
        with gzip.open(cached_path(DATASET_MAP[dataset_type])) as file:
            for line in file:
                review_json = json.loads(line)
                genres = review_json["genres"]
                for label, sentence in review_json["review_sentences"]:
                    data.append({"label": label, "sentence": sentence, "genre": genres})

        dataset = ListDataset(data)
        return DataLoader(
            dataset,
            num_workers=2,
            collate_fn=prepare_sample,
            batch_size=batch_size,
            shuffle=True,
        )


class GoodreadsMultiSentenceDataset(BaseDataset):
    def __init__(self, hparams):
        super().__init__()
        self._max_length = hparams.max_length

    def get_dataloader(self, dataset_type, tokenizer, batch_size):
        def prepare_sample(samples):
            genres = [encode_genre(x["genre"]) for x in samples]
            encoded_sentences = [
                tokenizer.batch_encode_plus(
                    s["sentences"], max_length=self._max_length, pad_to_max_length=True
                )
                for s in samples
            ]

            return (
                pad_sequence(
                    [tensor(es["input_ids"]) for es in encoded_sentences],
                    padding_value=0,
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

        data = []
        with gzip.open(cached_path(DATASET_MAP[dataset_type])) as file:
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
            num_workers=2,
            collate_fn=prepare_sample,
            batch_size=batch_size,
            shuffle=True,
        )


class GoodreadsSscDataset(BaseDataset):
    def __init__(self, hparams):
        super().__init__()
        self._max_length = hparams.max_length
        self._max_sent_per_example = 3

    def get_dataloader(self, dataset_type, tokenizer, batch_size):
        def prepare_sample(samples):
            sentences = ["[SEP]".join(x["sentences"]) for x in samples]

            output = tokenizer.batch_encode_plus(
                sentences, max_length=self._max_length, pad_to_max_length=True
            )
            input_ids = tensor(output["input_ids"])
            num_sentences = sum(
                sum(input_ids == tokenizer._convert_token_to_id("[SEP]"))
            )
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
                    max_length=self._max_sent_per_example,
                ),
                pad_sequence(
                    [tensor(x["labels"]) for x in samples],
                    max_length=self._max_sent_per_example,
                ),
            )

        data = []
        with gzip.open(cached_path(DATASET_MAP[dataset_type])) as file:
            for line in file:
                review_json = json.loads(line)
                genres = review_json["genres"]
                sentences, labels = list(), list()
                for label, sentence in review_json["review_sentences"]:
                    labels.append(label)
                    sentences.append(sentence)

                for (sentences_loop, labels_loop) in self.enforce__max_sent_per_example(
                    sentences, labels
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
            num_workers=2,
            collate_fn=prepare_sample,
            batch_size=batch_size,
            shuffle=True,
        )

    def enforce__max_sent_per_example(self, sentences, labels=None):
        """
        Splits examples with len(sentences) > self._max_sent_per_example into multiple smaller examples
        with len(sentences) <= self._max_sent_per_example.
        Recursively split the list of sentences into two halves until each half
        has len(sentences) < <= self._max_sent_per_example. The goal is to produce splits that are of almost
        equal size to avoid the scenario where all splits are of size
        self._max_sent_per_example then the last split is 1 or 2 sentences
        This will result into losing context around the edges of each examples.
        """
        if labels is not None:
            assert len(sentences) == len(labels)

        if (
            len(sentences) > self._max_sent_per_example
            and self._max_sent_per_example > 0
        ):
            i = len(sentences) // 2
            l1 = self.enforce__max_sent_per_example(
                sentences[:i], None if labels is None else labels[:i]
            )
            l2 = self.enforce__max_sent_per_example(
                sentences[i:], None if labels is None else labels[i:]
            )
            return l1 + l2
        else:
            return [(sentences, labels)]

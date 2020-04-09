import csv
import gzip
import itertools
import json
from functools import partial

from torch import tensor
from torch.utils.data import DataLoader
from transformers import cached_path

from spoiler_detection.datasets.base_dataset import BaseDataset, ListDataset
from spoiler_detection.datasets.datasets_maps import get_tvtropes_map
from spoiler_detection.datasets.utils import pad_sequence
from spoiler_detection.feature_encoders import encode_genre


class TvTropesMovieSingleSentenceDataset(BaseDataset):
    def __init__(self, hparams):
        super().__init__()
        self._max_length = hparams.max_length

    def prepare_sample(self, tokenizer, samples):
        sentences = [x["sentence"] for x in samples]
        labels = [x["label"] for x in samples]

        output = tokenizer.batch_encode_plus(
            sentences, max_length=self._max_length, pad_to_max_length=True
        )
        return (
            tensor(output["input_ids"]),
            tensor(output["attention_mask"]),
            tensor(output["token_type_ids"]),
            tensor([]),
            tensor(labels),
        )

    def get_dataloader(self, dataset_type, tokenizer, batch_size):
        data = []
        with open(cached_path(get_tvtropes_map()[dataset_type])) as file:
            reader = csv.reader(file)
            next(reader)  # skip header
            for sentence, spoiler, verb, page, trope in reader:
                label = 1 if spoiler == "True" else 0
                data.append({"label": label, "sentence": sentence})

        dataset = ListDataset(data)
        return DataLoader(
            dataset,
            num_workers=2,
            collate_fn=partial(self.prepare_sample, tokenizer),
            batch_size=batch_size,
            shuffle=True,
        )

from transformers import cached_path
from torch import tensor
from torch.utils.data import DataLoader, Dataset
from functools import partial
import json
import gzip
from spoiler_detection.feature_encoders import encode_genre

DATASET_MAP = {
    "train": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-train.json.gz",
    "val": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-val.json.gz",
    "test": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-test.json.gz",
    "dev": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads-lite.json.gz",
}


class ListDataset(Dataset):
    def __init__(self, data):
        super(ListDataset).__init__()
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class GoodreadsSingleSentenceDataset:
    def __init__(self, max_length=128):
        super().__init__()
        self._max_length = max_length

    def get_dataloader(self, dataset_type, tokenizer):
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
            dataset, num_workers=2, collate_fn=prepare_sample, batch_size=32,
        )

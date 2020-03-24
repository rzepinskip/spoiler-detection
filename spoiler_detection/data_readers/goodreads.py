from spoiler_detection.file_utils import cached_path
from torch import tensor
from torch.utils.data import DataLoader, Dataset
from functools import partial
import json


class TextLineDataset(Dataset):
    def __init__(self, path):
        super(TextLineDataset).__init__()
        self.data = []
        with open(path) as file:
            for line in file:
                review_json = json.loads(line)
                for label, sentence in review_json["review_sentences"]:
                    self.data.append({"label": label, "sentence": sentence})

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def prepare_sample(tokenizer, samples):
    max_length = 128
    sentences = [x["sentence"] for x in samples]
    labels = [x["label"] for x in samples]

    output = tokenizer.batch_encode_plus(
        sentences, max_length=max_length, pad_to_max_length=True
    )
    return (
        tensor(output["input_ids"]),
        tensor(output["attention_mask"]),
        tensor(output["token_type_ids"]),
        tensor(labels).long(),
    )


def get_goodreads_dataset(tokenizer, dataset_type):
    if dataset_type == "train":
        path = cached_path(
            "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-train.json"
        )
    elif dataset_type == "val":
        path = cached_path(
            "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-val.json"
        )
    elif dataset_type == "test":
        path = cached_path(
            "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-test.json"
        )
    elif dataset_type == "dev":
        path = cached_path(
            "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads-lite.json"
        )
    else:
        raise ValueError("Wrong dataset type")

    dataset = TextLineDataset(path)
    prepare_sample_partial = partial(prepare_sample, tokenizer)
    return DataLoader(
        dataset, num_workers=2, collate_fn=prepare_sample_partial, batch_size=32
    )

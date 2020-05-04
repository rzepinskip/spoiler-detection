import os
from argparse import ArgumentParser

from spoiler_detection.datasets import (
    GoodreadsSingleDataset,
    GoodreadsSingleGenreAppendedDataset,
    GoodreadsSscDataset,
)

model_types = ["bert-base-cased", "bert-base-uncased", "albert-base-v2", "roberta-base"]
dataset_classes = [GoodreadsSingleDataset]

for model_type in model_types:
    for dataset_class in dataset_classes:
        parser = ArgumentParser(add_help=False)

        parser.add_argument("--model_type", default=model_type, type=str)
        parser.add_argument("--max_length", default=128, type=int)
        parser.add_argument("--max_sentences", default=5, type=int)
        args = parser.parse_args([])

        dataset = dataset_class(hparams=args)
        dataset_types = ["train", "val"]
        for dataset_type in dataset_types:
            if not os.path.isfile(dataset.get_file_name(dataset_type)):
                print(
                    f"[{model_type}][{dataset_class.__name__}][{dataset_type}] Tokenizing..."
                )
                dataset.write_dataset(dataset_type)

import os
from argparse import ArgumentParser

from spoiler_detection.datasets import (
    GoodreadsSingleDataset,
    GoodreadsSingleGenreAppendedDataset,
    GoodreadsSscDataset,
    TvTropesBookSscDataset,
)

model_types = ["google/electra-base-discriminator"]
dataset_classes = [TvTropesBookSscDataset]
max_sentences_limits = [5]
max_length_limits = [512]

for model_type in model_types:
    for dataset_class in dataset_classes:
        for max_length in max_length_limits:
            for max_sentences in max_sentences_limits:
                parser = ArgumentParser(add_help=False)

                parser.add_argument("--model_type", default=model_type, type=str)
                parser.add_argument("--max_length", default=max_length, type=int)
                parser.add_argument("--max_sentences", default=max_sentences, type=int)
                args = parser.parse_args([])

                dataset = dataset_class(hparams=args)
                dataset_types = ["train", "val", "test"]
                for dataset_type in dataset_types:
                    if not os.path.isfile(dataset.get_file_name(dataset_type)):
                        print(
                            f"[{model_type}][{dataset_class.__name__}][{dataset_type}] Tokenizing..."
                        )
                        dataset.write_dataset(dataset_type)

import itertools
from argparse import ArgumentParser

import pytest
from tqdm import tqdm
from transformers import AutoTokenizer

from spoiler_detection.datasets import (
    GoodreadsMultiSentenceDataset,
    GoodreadsSingleSentenceDataset,
    GoodreadsSscDataset,
)

max_length = 128
batch_size = 32 * 100
dataset_type = GoodreadsSingleSentenceDataset

parser = ArgumentParser()
parser = dataset_type.add_dataset_specific_args(parser)
args = parser.parse_args([])
args.max_length = max_length
args.num_workers = 4

tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

dataset = dataset_type(args)
data_loader = dataset.get_dataloader("test", tokenizer, batch_size=batch_size)
it = iter(data_loader)
non_zero_count = 0
for batch in tqdm(it):
    input_ids = batch[0]
    last_tokens = input_ids[:, max_length - 1]
    non_zero_count += sum(last_tokens != 0).item()

samples = len(it) * batch_size
print(
    f"{non_zero_count} out of {samples} truncated ({non_zero_count / (samples) * 100:.4f}%)"
)

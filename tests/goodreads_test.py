from argparse import ArgumentParser

import pytest
from transformers import AutoTokenizer

from spoiler_detection.datasets import (
    GoodreadsMultiSentenceDataset,
    GoodreadsSingleSentenceDataset,
    GoodreadsSscDataset,
)


def test_GoodreadsSingleSentenceDataset():
    parser = ArgumentParser()
    parser = GoodreadsSingleSentenceDataset.add_dataset_specific_args(parser)
    args = parser.parse_args([])
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

    dataset = GoodreadsSingleSentenceDataset(args)
    data_loader = dataset.get_dataloader("dev", tokenizer, batch_size=2)
    x = list(data_loader)
    assert len(x) > 0


def test_GoodreadsMultiSentenceDataset():
    parser = ArgumentParser()
    parser = GoodreadsMultiSentenceDataset.add_dataset_specific_args(parser)
    args = parser.parse_args([])
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

    dataset = GoodreadsMultiSentenceDataset(args)
    data_loader = dataset.get_dataloader("dev", tokenizer, batch_size=2)
    x = list(data_loader)
    assert len(x) > 0


def test_GoodreadsSscDataset():
    parser = ArgumentParser()
    parser = GoodreadsSscDataset.add_dataset_specific_args(parser)
    args = parser.parse_args([])
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

    dataset = GoodreadsSscDataset(args)
    data_loader = dataset.get_dataloader("dev", tokenizer, batch_size=2)
    x = list(data_loader)
    assert len(x) > 0

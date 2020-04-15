import numpy as np
import torch

from spoiler_detection.datasets.utils import enforce_max_sent_per_example, pad_sequence


def test_pad_sequence():
    x = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    x_padded = pad_sequence(x, batch_first=True, padding_value=-1, max_length=4)
    assert torch.equal(x_padded, torch.tensor([[1, 2, 3, -1], [4, 5, -1, -1]]))


def test_enforce_max_sent_per_example():
    max_sentences = 4
    sentences, labels = (
        ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
        ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9",],
    )
    split = [
        (len(s), len(l))
        for s, l in enforce_max_sent_per_example(sentences, labels, max_sentences)
    ]
    assert split == [(3, 3), (3, 3), (3, 3)]

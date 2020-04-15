import numpy as np


def pad_sequence(sequences, batch_first=True, padding_value=-1, max_length=None):
    # Based on https://pytorch.org/docs/stable/_modules/torch/nn/utils/rnn.html#pad_sequence
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    if max_length is None:
        max_length = max([s.size(0) for s in sequences])

    if batch_first:
        out_dims = (len(sequences), max_length) + trailing_dims
    else:
        out_dims = (max_length, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


def enforce_max_sent_per_example(sentences, labels=None, max_sentences=0):
    if labels is not None:
        assert len(sentences) == len(labels)

    chunks = (
        len(sentences) // max_sentences
        if len(sentences) % max_sentences == 0
        else len(sentences) // max_sentences + 1
    )
    return zip(np.array_split(sentences, chunks), np.array_split(labels, chunks))


def enforce_max_sent_per_example_legacy(sentences, labels=None, max_sentences=0):
    """
    Splits examples with len(sentences) > max_sentences into multiple smaller examples
    with len(sentences) <= max_sentences.
    Recursively split the list of sentences into two halves until each half
    has len(sentences) < <= max_sentences. The goal is to produce splits that are of almost
    equal size to avoid the scenario where all splits are of size
    max_sentences then the last split is 1 or 2 sentences
    This will result into losing context around the edges of each examples.
    """
    if labels is not None:
        assert len(sentences) == len(labels)

    if len(sentences) > max_sentences and max_sentences > 0:
        i = len(sentences) // 2
        l1 = enforce_max_sent_per_example_legacy(
            sentences[:i], None if labels is None else labels[:i], max_sentences
        )
        l2 = enforce_max_sent_per_example_legacy(
            sentences[i:], None if labels is None else labels[i:], max_sentences
        )
        return l1 + l2
    else:
        return [(sentences, labels)]

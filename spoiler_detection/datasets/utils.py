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

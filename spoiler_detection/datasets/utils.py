import numpy as np


def get_model_group(model_type):
    if model_type.startswith("bert"):
        name_split = model_type.split("-")
        return f"{name_split[0]}_{name_split[2]}"

    model_group = model_type.split("-")[0]

    dash_sep = model_type.find("/")
    if dash_sep:
        model_group = model_group[dash_sep + 1 :]

    return model_group


def encode(texts, tokenizer, max_length=512):
    input_ids = tokenizer.batch_encode_plus(
        texts,
        return_attention_masks=False,
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=max_length,
    )["input_ids"]

    return np.array(input_ids)


def enforce_max_sent_per_example(sentences, labels, max_sentences=1):
    assert len(sentences) == len(labels)

    chunks = (
        len(sentences) // max_sentences
        if len(sentences) % max_sentences == 0
        else len(sentences) // max_sentences + 1
    )
    return zip(np.array_split(sentences, chunks), np.array_split(labels, chunks))

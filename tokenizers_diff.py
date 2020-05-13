from transformers import AutoModel, AutoTokenizer

roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "A Titan RTX has 24GB of VRAM"
print(roberta_tokenizer.tokenize(sequence))
print(bert_tokenizer.tokenize(sequence))

seq = "The first sentence to tokenize."
seq_pair = "The second sentence to tokenize."
print(roberta_tokenizer.encode_plus(text=seq, text_pair=seq_pair))
print(bert_tokenizer.encode_plus(text=seq, text_pair=seq_pair))

# AutoModel.from_pretrained("google/electra-base-discriminator")
print(bert_tokenizer.batch_encode_plus([seq]))
print(bert_tokenizer.batch_encode_plus([(seq, seq_pair)]))


def get_model_group(model_type):
    if model_type.startswith("bert"):
        name_split = model_type.split("-")
        return f"{name_split[0]}_{name_split[2]}"

    model_group = model_type.split("-")[0]

    dash_sep = model_type.find("/")
    if dash_sep:
        model_group = model_group[dash_sep + 1 :]

    return model_group


print(get_model_group("google/electra-base-discriminator"))

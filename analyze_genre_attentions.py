import pickle

import numpy as np
from tqdm import tqdm

with open("attentions.pickle", "rb") as handle:
    read_results = pickle.load(handle)


attention_dist = []
for label, tokens, attention in tqdm(read_results):
    sentence_end, tokens_end = np.where(tokens == 102)[0]
    attention[0] = 0  # CLS
    attention[sentence_end] = 0  # SEP 1
    attention[tokens_end] = 0  # SEP 2
    attention_sum = attention.sum() if attention.sum() != 0 else attention
    attention = attention / attention_sum
    genre_attention = attention[sentence_end + 1 : tokens_end].sum()
    sentence_attention = attention[1:sentence_end].sum()
    uniform_genre_attention = (1 / (tokens_end + 1)) * (tokens_end - sentence_end - 1)
    attention_dist += [
        (label, sentence_attention, genre_attention, uniform_genre_attention)
    ]

mean_genre_attention = np.mean([x for label, _, x, _ in attention_dist])
mean_uniform_genre_attention = np.mean([x for label, _, _, x in attention_dist])
print(f"[All] Real {mean_genre_attention} vs uniform {mean_uniform_genre_attention}")

mean_genre_attention = np.mean([x for label, _, x, _ in attention_dist if label == 0.0])
mean_uniform_genre_attention = np.mean(
    [x for label, _, _, x in attention_dist if label == 0.0]
)
print(
    f"[Label 0] Real {mean_genre_attention} vs uniform {mean_uniform_genre_attention}"
)

mean_genre_attention = np.mean([x for label, _, x, _ in attention_dist if label == 1.0])
mean_uniform_genre_attention = np.mean(
    [x for label, _, _, x in attention_dist if label == 1.0]
)
print(
    f"[Label 1] Real {mean_genre_attention} vs uniform {mean_uniform_genre_attention}"
)

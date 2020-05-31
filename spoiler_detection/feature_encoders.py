
from typing import Dict, List

encoding_dict = {
    "children": 0,
    "comics, graphic": 1,
    "fantasy, paranormal": 2,
    "fiction": 3,
    "history, historical fiction, biography": 4,
    "mystery, thriller, crime": 5,
    "non-fiction": 6,
    "poetry": 7,
    "romance": 8,
    "young-adult": 9,
}


def encode_as_distribution(genres: Dict[str, int]) -> List[float]:
    feature = [0] * len(encoding_dict)
    for genre_name, count in genres.items():
        feature[encoding_dict[genre_name]] = count

    normalized_feature = [x / sum(feature) for x in feature]
    return normalized_feature


simplification_dict = {
    "children": "children",
    "comics, graphic": "comics",
    "fantasy, paranormal": "fantasy",
    "fiction": "fiction",
    "history, historical fiction, biography": "history",
    "mystery, thriller, crime": "mystery",
    "non-fiction": "fact", # avoid negation
    "poetry": "poetry",
    "romance": "romance",
    "young-adult": "youth", # avoid multiword
}

def encode_as_string(genres: Dict[str, int]) -> str:
    return ", ".join([simplification_dict[k] for k, v in sorted(genres.items(), key=lambda item: item[1])])

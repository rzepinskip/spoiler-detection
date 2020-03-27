from typing import Dict, List

encoding_dict = {
    "children": 0,
    "comics, graphic": 1,
    "fantasy, paranormal": 2,
    "fiction": 3,
    "history, historical fiction, biography": 3,
    "mystery, thriller, crime": 4,
    "non-fiction": 5,
    "poetry": 6,
    "romance": 7,
    "young-adult": 8,
}


def encode_genre(genres: Dict[str, int]) -> List[float]:
    feature = [0] * len(encoding_dict)
    for genre_name, count in genres.items():
        feature[encoding_dict[genre_name]] = count

    normalized_feature = [x / sum(feature) for x in feature]
    return normalized_feature

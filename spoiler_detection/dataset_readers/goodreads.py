from typing import Dict, List, Optional
import json
import logging
import numpy as np

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from spoiler_detection.dataset_readers.readers import (
    SingleSentenceDatasetReader,
    MultipleSentencesDatasetReader,
)
from spoiler_detection.feature_encoders import encode_genre

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("goodreads_single_sentence")
class GoodreadsSingleSentenceDatasetReader(SingleSentenceDatasetReader):
    def __init__(
        self,
        lazy: bool = False,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        cache_directory: Optional[str] = None,
    ) -> None:
        super().__init__(lazy, tokenizer, token_indexers, cache_directory)

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                review_json = json.loads(line)
                for is_spoiler, sentence in review_json["review_sentences"]:
                    instance = self.text_to_instance(sentence, is_spoiler)
                    instance.add_field(
                        "genre",
                        ArrayField(np.array(encode_genre(review_json["genres"]))),
                    )
                    yield instance


@DatasetReader.register("goodreads_multiple_sentences")
class GoodreadsMultipleSentencesDatasetReader(MultipleSentencesDatasetReader):
    def __init__(
        self,
        lazy: bool = False,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        cache_directory: Optional[str] = None,
    ) -> None:
        super().__init__(lazy, tokenizer, token_indexers, cache_directory)

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                review_json = json.loads(line)
                sentences, labels = list(), list()
                for is_spoiler, sentence in review_json["review_sentences"]:
                    sentences.append(sentence)
                    labels.append(int(is_spoiler))

                instance = self.text_to_instance(sentences, labels)
                instance.add_field(
                    "genre", ArrayField(np.array(encode_genre(review_json["genres"]))),
                )
                yield instance

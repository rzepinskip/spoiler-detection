from typing import Dict, List, Optional
import json
import logging
import numpy as np
import gzip
import itertools

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data import Instance
from allennlp.data.fields import Field, TextField, LabelField, ListField, ArrayField
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from spoiler_detection.dataset_readers.readers import (
    SingleSentenceDatasetReader,
    MultipleSentencesDatasetReader,
)
from spoiler_detection.feature_encoders import encode_genre

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def open_file(path, mode):
    if path.endswith("json") or path.endswith("jsonl"):
        return open(path, mode)
    else:
        return gzip.open(path, mode)


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
        with open_file(cached_path(file_path), "r") as data_file:
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
        with open_file(cached_path(file_path), "r") as data_file:
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


@DatasetReader.register("goodreads_multiple_sentences-ssc")
class GoodreadsSscMultipleSentencesDatasetReader(MultipleSentencesDatasetReader):
    def __init__(
        self,
        lazy: bool = False,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        cache_directory: Optional[str] = None,
        max_sent_per_example: int = 3,  # TODO high numbers cause error with tensor sizes
    ) -> None:
        super().__init__(lazy, tokenizer, token_indexers, cache_directory)
        self.max_sent_per_example = max_sent_per_example

    @overrides
    def _read(self, file_path: str):
        with open_file(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                review_json = json.loads(line)
                sentences, labels = list(), list()
                for is_spoiler, sentence in review_json["review_sentences"]:
                    sentences.append(sentence)
                    labels.append(int(is_spoiler))

                for (sentences_loop, labels_loop) in self.enforce_max_sent_per_example(
                    sentences, labels
                ):
                    instance = self.text_to_instance(sentences_loop, labels_loop)
                    instance.add_field(
                        "genre",
                        ArrayField(np.array(encode_genre(review_json["genres"]))),
                    )
                    yield instance

    def enforce_max_sent_per_example(self, sentences, labels=None):
        """
        Splits examples with len(sentences) > self.max_sent_per_example into multiple smaller examples
        with len(sentences) <= self.max_sent_per_example.
        Recursively split the list of sentences into two halves until each half
        has len(sentences) < <= self.max_sent_per_example. The goal is to produce splits that are of almost
        equal size to avoid the scenario where all splits are of size
        self.max_sent_per_example then the last split is 1 or 2 sentences
        This will result into losing context around the edges of each examples.
        """
        if labels is not None:
            assert len(sentences) == len(labels)

        if len(sentences) > self.max_sent_per_example and self.max_sent_per_example > 0:
            i = len(sentences) // 2
            l1 = self.enforce_max_sent_per_example(
                sentences[:i], None if labels is None else labels[:i]
            )
            l2 = self.enforce_max_sent_per_example(
                sentences[i:], None if labels is None else labels[i:]
            )
            return l1 + l2
        else:
            return [(sentences, labels)]

    def text_to_instance(
        self, sentences: List[str], labels: List[str] = None
    ) -> Instance:
        fields: Dict[str, Field] = {}
        tokenized_sentences = [
            self._tokenizer.tokenize(s) + [Token("[SEP]", text_id=102)]
            for s in sentences
        ]
        concatenated_sentences = [Token("[CLS]", text_id=101)] + list(
            itertools.chain.from_iterable(tokenized_sentences)
        )
        fields["sentences"] = TextField(concatenated_sentences, self._token_indexers)

        if labels is not None:
            fields["labels"] = ListField(
                [LabelField(label, skip_indexing=True) for label in labels]
            )
        return Instance(fields)

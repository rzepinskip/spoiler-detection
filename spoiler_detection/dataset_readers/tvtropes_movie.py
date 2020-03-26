from typing import Dict, List, Optional
import logging
import csv

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    Field,
    LabelField,
    TextField,
    ListField,
    SequenceLabelField,
)
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from spoiler_detection.dataset_readers.readers import (
    SingleSentenceDatasetReader,
    MultipleSentencesDatasetReader,
)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("tvtropes_movie_single_sentence")
class TvTropesMovieSingleSentenceDatasetReader(SingleSentenceDatasetReader):
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
            reader = csv.reader(data_file)
            next(reader)  # skip header
            for sentence, spoiler, verb, page, trope in reader:
                yield self.text_to_instance(sentence, 1 if spoiler == "True" else 0)


@DatasetReader.register("tvtropes_movie_multiple_sentences")
class TvTropesMovieMultipleSentencesDatasetReader(MultipleSentencesDatasetReader):
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
            reader = csv.reader(data_file)
            next(reader)  # skip header
            for sentence, spoiler, verb, page, trope in reader:
                yield self.text_to_instance([sentence], [1 if spoiler == "True" else 0])

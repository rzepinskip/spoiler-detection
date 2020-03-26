from typing import Dict, List, Optional
from overrides import overrides

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


class SingleSentenceDatasetReader(DatasetReader):
    def __init__(
        self,
        lazy: bool = False,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        cache_directory: Optional[str] = None,
    ) -> None:
        super().__init__(lazy, cache_directory)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def text_to_instance(self, sentence: str, label: str = None) -> Instance:
        fields: Dict[str, Field] = {}
        tokenized_sentence = self._tokenizer.tokenize(sentence)
        fields["sentence"] = TextField(tokenized_sentence, self._token_indexers)
        if label is not None:
            fields["label"] = LabelField(label, skip_indexing=True)
        return Instance(fields)


class MultipleSentencesDatasetReader(DatasetReader):
    def __init__(
        self,
        lazy: bool = False,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        cache_directory: Optional[str] = None,
    ) -> None:
        super().__init__(lazy, cache_directory)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def text_to_instance(
        self, sentences: List[str], labels: List[str] = None
    ) -> Instance:
        fields: Dict[str, Field] = {}
        tokenized_sentences = [self._tokenizer.tokenize(sent) for sent in sentences]
        sentence_sequence = ListField(
            [TextField(tk, self._token_indexers) for tk in tokenized_sentences]
        )
        fields["sentences"] = sentence_sequence

        if labels is not None:
            fields["labels"] = SequenceLabelField(labels, sentence_sequence)
        return Instance(fields)

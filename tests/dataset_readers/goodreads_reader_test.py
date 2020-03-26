# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from spoiler_detection.dataset_readers import (
    GoodreadsSingleSentenceDatasetReader,
    GoodreadsMultipleSentencesDatasetReader,
)


class TestGoodreadsSingleSentenceDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):

        reader = GoodreadsSingleSentenceDatasetReader()
        instances = ensure_list(reader.read("tests/fixtures/goodreads.jsonl"))

        instance1 = {
            "sentence": [
                "A",
                "fun",
                ",",
                "fast",
                "paced",
                "science",
                "fiction",
                "thriller",
                ".",
            ],
            "label": 0,
        }

        instance2 = {
            "sentence": [
                "It",
                "is",
                "a",
                "book",
                "about",
                "choice",
                "and",
                "regret",
                ".",
            ],
            "label": 1,
        }

        assert len(instances) == 156
        fields = instances[23].fields
        assert [t.text for t in fields["sentence"].tokens] == instance1["sentence"]
        assert fields["label"].label == instance1["label"]
        fields = instances[35].fields
        assert [t.text for t in fields["sentence"].tokens] == instance2["sentence"]
        assert fields["label"].label == instance2["label"]


class TestGoodreadsMultipleSentencesDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):

        reader = GoodreadsMultipleSentencesDatasetReader()
        instances = ensure_list(reader.read("tests/fixtures/goodreads.jsonl"))

        instance1_sentence1 = {
            "sentence": [
                "A",
                "fun",
                ",",
                "fast",
                "paced",
                "science",
                "fiction",
                "thriller",
                ".",
            ],
            "label": 0,
        }

        instance1_sentence2 = {
            "sentence": [
                "It",
                "is",
                "a",
                "book",
                "about",
                "choice",
                "and",
                "regret",
                ".",
            ],
            "label": 1,
        }

        assert len(instances) == 10
        fields = instances[2].fields
        assert [t.text for t in fields["sentences"][0].tokens] == instance1_sentence1[
            "sentence"
        ]
        assert fields["labels"][0] == instance1_sentence1["label"]
        assert [t.text for t in fields["sentences"][12].tokens] == instance1_sentence2[
            "sentence"
        ]
        assert fields["labels"][12] == instance1_sentence2["label"]

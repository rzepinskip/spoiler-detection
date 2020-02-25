# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from spoiler_detection.dataset_readers import GoodreadsSingleSentenceDatasetReader


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
            "label": "nonspoiler",
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
            "label": "spoiler",
        }

        assert len(instances) == 156
        fields = instances[23].fields
        assert [t.text for t in fields["sentence"].tokens] == instance1["sentence"]
        assert fields["label"].label == instance1["label"]
        fields = instances[35].fields
        assert [t.text for t in fields["sentence"].tokens] == instance2["sentence"]
        assert fields["label"].label == instance2["label"]


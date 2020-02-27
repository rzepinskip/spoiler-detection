# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from spoiler_detection.dataset_readers import (
    TvTropesMovieSingleSentenceDatasetReader,
    TvTropesMovieMultipleSentencesDatasetReader,
)


class TestTvTropesMovieSingleSentenceDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):

        reader = TvTropesMovieSingleSentenceDatasetReader()
        instances = ensure_list(reader.read("tests/fixtures/tvtropes_movie.csv"))

        instance1 = {
            "sentence": [
                "Because",
                "of",
                "Never",
                "Say",
                '"',
                "Die",
                '"',
                ",",
                "a",
                "horrifying",
                "scene",
                "became",
                "arguably",
                "EVEN",
                "WORSE",
                ".",
            ],
            "label": "spoiler",
        }

        instance2 = {
            "sentence": [
                "The",
                "end",
                "of",
                '"',
                "Mutant",
                "Rain",
                '"',
                "has",
                "him",
                "reveal",
                "his",
                "face",
                "to",
                "Aaron",
                "and",
                "the",
                "rest",
                "of",
                "his",
                "team",
                ".",
            ],
            "label": "nonspoiler",
        }

        assert len(instances) == 10
        fields = instances[0].fields
        assert [t.text for t in fields["sentence"].tokens] == instance1["sentence"]
        assert fields["label"].label == instance1["label"]
        fields = instances[3].fields
        assert [t.text for t in fields["sentence"].tokens] == instance2["sentence"]
        assert fields["label"].label == instance2["label"]


class TestTvTropesMovieMultipleSentencesDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):

        reader = TvTropesMovieMultipleSentencesDatasetReader()
        instances = ensure_list(reader.read("tests/fixtures/tvtropes_movie.csv"))

        instance1 = {
            "sentence": [
                "Because",
                "of",
                "Never",
                "Say",
                '"',
                "Die",
                '"',
                ",",
                "a",
                "horrifying",
                "scene",
                "became",
                "arguably",
                "EVEN",
                "WORSE",
                ".",
            ],
            "label": "spoiler",
        }

        instance2 = {
            "sentence": [
                "The",
                "end",
                "of",
                '"',
                "Mutant",
                "Rain",
                '"',
                "has",
                "him",
                "reveal",
                "his",
                "face",
                "to",
                "Aaron",
                "and",
                "the",
                "rest",
                "of",
                "his",
                "team",
                ".",
            ],
            "label": "nonspoiler",
        }

        assert len(instances) == 10
        fields = instances[0].fields
        assert [t.text for t in fields["sentences"][0].tokens] == instance1["sentence"]
        assert fields["labels"][0] == instance1["label"]
        fields = instances[3].fields
        assert [t.text for t in fields["sentences"][0].tokens] == instance2["sentence"]
        assert fields["labels"][0] == instance2["label"]

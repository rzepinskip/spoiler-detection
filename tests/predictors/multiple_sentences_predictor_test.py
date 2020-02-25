# pylint: disable=no-self-use,invalid-name,unused-import
from unittest import TestCase

from pytest import approx
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

# required so that our custom model + predictor + dataset reader
# will be registered by name
import spoiler_detection


class TestMultipleSentencesPredictor(TestCase):
    def test_uses_named_inputs(self):
        inputs = {"sentences": ["It can be a spoiler", "Or not."]}

        archive = load_archive("tests/fixtures/multiple_sentences_classifier.tar.gz")
        predictor = Predictor.from_archive(archive, "multiple_sentences_predictor")

        instance = predictor._json_to_instance(inputs)
        result = predictor.predict_instance(instance)

        labels = result.get("labels")
        for label in labels:
            assert label in {"nonspoiler", "spoiler"}

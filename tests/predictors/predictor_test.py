# pylint: disable=no-self-use,invalid-name,unused-import
from unittest import TestCase

from pytest import approx
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

# required so that our custom model + predictor + dataset reader
# will be registered by name
import spoiler_detection


class TestTextClassifierPredictor(TestCase):
    def test_uses_named_inputs(self):
        inputs = {"sentence": "It can be a spoiler."}

        archive = load_archive("tests/fixtures/text_classifier_model.tar.gz")
        predictor = Predictor.from_archive(archive, "text-classifier")

        result = predictor.predict_json(inputs)

        label = result.get("label")
        assert label in {"nonspoiler", "spoiler"}

        class_probabilities = result.get("class_probabilities")
        assert class_probabilities is not None
        assert all(cp > 0 for cp in class_probabilities)
        assert sum(class_probabilities) == approx(1.0)

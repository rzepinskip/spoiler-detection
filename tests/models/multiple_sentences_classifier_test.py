# pylint: disable=invalid-name,protected-access
from allennlp.common.testing import ModelTestCase
import spoiler_detection


class MultpipleSentencesClassifierTest(ModelTestCase):
    def setUp(self):
        super(MultpipleSentencesClassifierTest, self).setUp()
        self.set_up_model(
            "tests/fixtures/multiple_sentences_classifier.json",
            "tests/fixtures/goodreads.jsonl",
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

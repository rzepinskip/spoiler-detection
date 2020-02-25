# pylint: disable=invalid-name,protected-access
from allennlp.common.testing import ModelTestCase
import spoiler_detection


class SingleSentenceClassifierTest(ModelTestCase):
    def setUp(self):
        super(SingleSentenceClassifierTest, self).setUp()
        self.set_up_model(
            "tests/fixtures/single_sentence_classifier.json",
            "tests/fixtures/goodreads.jsonl",
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

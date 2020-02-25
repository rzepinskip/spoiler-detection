from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register("multiple_sentences_predictor")
class MultipleSentencesPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentences = json_dict["sentences"]
        return self._dataset_reader.text_to_instance(sentences=sentences)

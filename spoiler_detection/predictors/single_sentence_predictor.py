from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register("single_sentence_predictor")
class SingleSentencePredictor(Predictor):
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        # label_dict will be like {0: "nonspoiler", 1: "spoiler"}
        label_dict = self._model.vocab.get_index_to_token_vocabulary("labels")
        # Convert it to list ["nonspoiler", "spoiler"]
        all_labels = [label_dict[i] for i in range(len(label_dict))]
        output_dict["all_labels"] = all_labels
        return output_dict

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence=sentence)

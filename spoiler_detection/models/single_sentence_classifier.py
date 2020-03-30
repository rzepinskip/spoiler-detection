from typing import Dict, Optional, List

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import (
    FeedForward,
    Seq2VecEncoder,
    TextFieldEmbedder,
)
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, Auc


@Model.register("single_sentence_classifier")
class SingleSentenceClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        feedforward: Optional[FeedForward] = None,
        dropout: float = None,
        class_weights: List[float] = None,
        use_genres: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder

        self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward
        if feedforward is not None:
            self._classifier_input_dim = self._feedforward.get_output_dim()
        else:
            self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._classification_layer = torch.nn.Linear(
            self._classifier_input_dim, 2  # number of labels
        )
        self._metrics = {
            "accuracy": CategoricalAccuracy(),
            "f1": F1Measure(positive_label=1),
            "auc": Auc(positive_label=1),
        }
        if class_weights is not None:
            self._loss = torch.nn.CrossEntropyLoss(
                weight=torch.FloatTensor(class_weights)
            )
        else:
            self._loss = torch.nn.CrossEntropyLoss()

        self.use_genres = use_genres

        initializer(self)

    def forward(  # type: ignore
        self,
        sentence: TextFieldTensors,
        label: torch.IntTensor = None,
        genre: torch.FloatTensor = None,
    ) -> Dict[str, torch.Tensor]:
        embedded_text = self._text_field_embedder(sentence)
        mask = get_text_field_mask(sentence)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self.use_genres:
            embedded_text = torch.cat((embedded_text, genre), dim=-1)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            for metric in self._metrics.values():
                if isinstance(metric, Auc):
                    metric(probs[:, 1], label)
                else:
                    metric(probs, label)

        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(
                self._label_namespace
            ).get(label_idx, str(label_idx))
            classes.append(label_str)
        output_dict["label"] = classes
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "f1": self._metrics["f1"].get_metric(reset=reset)[2],
            "accuracy": self._metrics["accuracy"].get_metric(reset=reset),
            "auc": self._metrics["auc"].get_metric(reset=reset),
        }

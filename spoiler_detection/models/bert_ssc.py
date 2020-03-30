from typing import Dict, Optional, List
from overrides import overrides
import torch
import logging

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

logger = logging.getLogger(__name__)


@Model.register("bert-ssc")
class BertSequential(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        dropout: float = None,
        class_weights: List[float] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder

        self._classifier_input_dim = self._text_field_embedder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self.num_labels = 2
        self._classification_layer = torch.nn.Linear(
            self._classifier_input_dim, self.num_labels  # number of labels
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

        initializer(self)

    def forward(  # type: ignore
        self,
        sentences: TextFieldTensors,  # sent_1, SEP, sent_2, SEP, ...
        labels: torch.IntTensor = None,
        genre: torch.FloatTensor = None,
    ) -> Dict[str, torch.Tensor]:
        embedded_sentences = self._text_field_embedder(sentences)

        if self._dropout:
            embedded_sentences = self._dropout(embedded_sentences)

        sentences_mask = (
            sentences["tokens"]["token_ids"] == 102
        ) 
        embedded_sentences = embedded_sentences[
            sentences_mask
        ]  
        assert embedded_sentences.dim() == 2
        num_sentences = embedded_sentences.shape[0]

        batch_size = 1
        embedded_sentences = embedded_sentences.unsqueeze(dim=0)
        embedded_sentences = self._dropout(embedded_sentences)

        if labels is not None:
            labels_mask = (
                labels != -1
            )  # mask for all the labels in the batch (no padding)

            labels = labels[
                labels_mask
            ]  # given batch_size x num_sentences_per_example return num_sentences_per_batch
            assert labels.dim() == 1

            num_labels = labels.shape[0]
            if (
                num_labels != num_sentences
            ):  # bert truncates long sentences, so some of the SEP tokens might be gone
                assert (
                    num_labels > num_sentences
                )  # but `num_labels` should be at least greater than `num_sentences`
                logger.warning(
                    f"Found {num_labels} labels but {num_sentences} sentences"
                )
                labels = labels[
                    :num_sentences
                ]  # Ignore some labels. This is ok for training but bad for testing.
                # We are ignoring this problem for now.
                # TODO: fix, at least for testing

            # similar to `embedded_sentences`, add an additional dimension that corresponds to batch_size=1
            labels = labels.unsqueeze(dim=0)

        # END: SEP tokens

        logits = self._classification_layer(embedded_sentences)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}
        loss = self._loss(logits.squeeze(dim=0), labels.squeeze(dim=0))
        output_dict["loss"] = loss
        for metric in self._metrics.values():
            if isinstance(metric, Auc):
                metric(probs[:, :, 1].flatten(), labels.flatten())
            else:
                metric(probs, labels)

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

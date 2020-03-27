from typing import Dict, Optional, List

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import (
    FeedForward,
    Seq2VecEncoder,
    TextFieldEmbedder,
    TimeDistributed,
    ConditionalRandomField,
)
from allennlp.nn import InitializerApplicator

from allennlp.training.metrics import CategoricalAccuracy, F1Measure


@Model.register("multiple_sentences_classifier")
class MultipleSentencesClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        feedforward: Optional[FeedForward] = None,
        dropout: float = None,
        class_weights: List[float] = None,
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

        self._metrics = {
            "accuracy": CategoricalAccuracy(),
            "f1": F1Measure(positive_label=1),
        }
        if class_weights is not None:
            self._loss = torch.nn.CrossEntropyLoss(
                weight=torch.FloatTensor(class_weights)
            )
        else:
            self._loss = torch.nn.CrossEntropyLoss()

        self._num_classes = 2
        self._classification_layer = TimeDistributed(
            torch.nn.Linear(self._classifier_input_dim, self._num_classes)
        )

        constraints = None  # allowed_transitions(label_encoding, labels)
        self._crf = ConditionalRandomField(
            self._num_classes, constraints, include_start_end_transitions=False
        )

        initializer(self)

    def forward(  # type: ignore
        self, sentences, labels
    ) -> Dict[str, torch.Tensor]:
        def get_text_field_mask(
            text_field_tensors: Dict[str, Dict[str, torch.Tensor]],
            num_wrapping_dims: int = 0,
        ) -> torch.BoolTensor:
            tensor_dims = [
                (tensor.dim(), tensor)
                for indexer_output in text_field_tensors.values()
                for tensor in indexer_output.values()
            ]
            tensor_dims.sort(key=lambda x: x[0])

            smallest_dim = tensor_dims[0][0] - num_wrapping_dims
            if smallest_dim == 2:
                token_tensor = tensor_dims[0][1]
                return token_tensor != 0
            elif smallest_dim == 3:
                character_tensor = tensor_dims[0][1]
                return (character_tensor > 0).any(dim=-1)
            else:
                raise ValueError(
                    "Expected a tensor with dimension 2 or 3, found {}".format(
                        smallest_dim
                    )
                )

        embeddings = []
        for batch_idx in range(sentences["tokens"]["token_ids"].size()[0]):
            embeddings.append(
                self._text_field_embedder(
                    {
                        "tokens": {
                            "token_ids": sentences["tokens"]["token_ids"][batch_idx],
                            "mask": sentences["tokens"]["mask"][batch_idx],
                            "type_ids": sentences["tokens"]["type_ids"][batch_idx],
                        }
                    }
                )
            )

        embedded_sentences = torch.stack(embeddings)
        token_masks = get_text_field_mask(sentences, 1)
        sentence_masks = get_text_field_mask(sentences)

        # get sentence embedding
        encoded_sentences = []
        n_sents = embedded_sentences.size()[
            1
        ]  # size: (n_batch, n_sents, n_tokens, n_embedding)
        for i in range(n_sents):
            embedded_text = self._seq2vec_encoder(
                embedded_sentences[:, i, :, :], mask=token_masks[:, i, :]
            )

            # TODO apply this layers on stacked sentences output somehow
            # if self._dropout:
            #     embedded_text = self._dropout(embedded_text)

            # if self._feedforward is not None:
            #     embedded_text = self._feedforward(embedded_text)

            encoded_sentences.append(embedded_text)

        encoded_sentences = torch.stack(
            encoded_sentences, 1
        )  # size: (n_batch, n_sents, n_embedding)

        # CRF prediction
        logits = self._classification_layer(
            encoded_sentences
        )  # size: (n_batch, n_sents, n_classes)
        best_paths = self._crf.viterbi_tags(logits, sentence_masks)
        predicted_labels = [x for x, y in best_paths]

        output_dict = {
            "logits": logits,
            "mask": sentence_masks,
            "labels": predicted_labels,
        }

        # referring to https://github.com/allenai/allennlp/blob/master/allennlp/models/crf_tagger.py#L229-L239
        if labels is not None:
            log_likelihood = self._crf(logits, labels, sentence_masks)
            output_dict["loss"] = -log_likelihood

            class_probabilities = logits * 0.0
            for i, instance_labels in enumerate(predicted_labels):
                for j, label_id in enumerate(instance_labels):
                    class_probabilities[i, j, label_id] = 1

            for metric in self._metrics.values():
                metric(class_probabilities, labels, sentence_masks)

        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        output_dict["labels"] = [
            [
                self.vocab.get_token_from_index(label, namespace="labels")
                for label in instance_labels
            ]
            for instance_labels in output_dict["labels"]
        ]
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "f1": self._metrics["f1"].get_metric(reset=reset)[2],
            "accuracy": self._metrics["accuracy"].get_metric(reset=reset),
        }

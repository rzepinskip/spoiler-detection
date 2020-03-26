from typing import Dict, Optional, List, Any

import numpy as np
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import (
    Seq2VecEncoder,
    TimeDistributed,
    TextFieldEmbedder,
    ConditionalRandomField,
    FeedForward,
)
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
# from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import F1Measure, CategoricalAccuracy


def get_text_field_mask(
    text_field_tensors: Dict[str, Dict[str, torch.Tensor]], num_wrapping_dims: int = 0
) -> torch.BoolTensor:
    """
    Takes the dictionary of tensors produced by a `TextField` and returns a mask
    with 0 where the tokens are padding, and 1 otherwise.  We also handle `TextFields`
    wrapped by an arbitrary number of `ListFields`, where the number of wrapping `ListFields`
    is given by `num_wrapping_dims`.

    If `num_wrapping_dims == 0`, the returned mask has shape `(batch_size, num_tokens)`.
    If `num_wrapping_dims > 0` then the returned mask has `num_wrapping_dims` extra
    dimensions, so the shape will be `(batch_size, ..., num_tokens)`.

    There could be several entries in the tensor dictionary with different shapes (e.g., one for
    word ids, one for character ids).  In order to get a token mask, we use the tensor in
    the dictionary with the lowest number of dimensions.  After subtracting `num_wrapping_dims`,
    if this tensor has two dimensions we assume it has shape `(batch_size, ..., num_tokens)`,
    and use it for the mask.  If instead it has three dimensions, we assume it has shape
    `(batch_size, ..., num_tokens, num_features)`, and sum over the last dimension to produce
    the mask.  Most frequently this will be a character id tensor, but it could also be a
    featurized representation of each token, etc.

    If the input `text_field_tensors` contains the "mask" key, this is returned instead of inferring the mask.
    """
    masks = []
    for indexer_name, indexer_tensors in text_field_tensors.items():
        if "mask" in indexer_tensors:
            masks.append(indexer_tensors["mask"].bool())
    if len(masks) == 1:
        return masks[0]
    elif len(masks) > 1:
        # TODO(mattg): My guess is this will basically never happen, so I'm not writing logic to
        # handle it.  Should be straightforward to handle, though.  If you see this error in
        # practice, open an issue on github.
        raise ValueError("found two mask outputs; not sure which to use!")

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
            "Expected a tensor with dimension 2 or 3, found {}".format(smallest_dim)
        )


@Model.register("multiple_sentences_classifier-old")
class MultipleSentencesClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        sentence_encoder: Seq2VecEncoder,
        initializer: InitializerApplicator = InitializerApplicator(),
        dropout: Optional[float] = None,
        regularizer: Optional[RegularizerApplicator] = None,
        class_weights: List[float] = None,
    ) -> None:
        super(MultipleSentencesClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.sentence_encoder = sentence_encoder
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "f1": F1Measure(positive_label=1),
        }

        if class_weights is not None:
            self.loss = torch.nn.CrossEntropyLoss(
                weight=torch.FloatTensor(class_weights)
            )
        else:
            self.loss = torch.nn.CrossEntropyLoss()

        self.label_projection_layer = TimeDistributed(
            Linear(self.sentence_encoder.get_output_dim(), 2)
        )

        constraints = None  # allowed_transitions(label_encoding, labels)
        self.crf = ConditionalRandomField(
            2, constraints, include_start_end_transitions=False
        )
        initializer(self)

    @overrides
    def forward(
        self, sentences: Dict[str, torch.LongTensor], labels: torch.LongTensor = None
    ) -> Dict[str, torch.Tensor]:

        # print(sentences['tokens'].size())
        # print(labels.size())

        embedded_sentences = self.text_field_embedder(sentences)
        token_masks = get_text_field_mask(sentences, 1)
        sentence_masks = get_text_field_mask(sentences)

        # get sentence embedding
        encoded_sentences = []
        n_sents = embedded_sentences.size()[
            1
        ]  # size: (n_batch, n_sents, n_tokens, n_embedding)
        for i in range(n_sents):
            encoded_sentences.append(
                self.sentence_encoder(
                    embedded_sentences[:, i, :, :], token_masks[:, i, :]
                )
            )
        encoded_sentences = torch.stack(encoded_sentences, 1)

        # dropout layer
        if self.dropout:
            encoded_sentences = self.dropout(encoded_sentences)

        # print(encoded_sentences.size()) # size: (n_batch, n_sents, n_embedding)

        # CRF prediction
        logits = self.label_projection_layer(
            encoded_sentences
        )  # size: (n_batch, n_sents, n_classes)
        best_paths = self.crf.viterbi_tags(logits, sentence_masks)
        predicted_labels = [x for x, y in best_paths]

        output_dict = {
            "logits": logits,
            "mask": sentence_masks,
            "labels": predicted_labels,
        }

        # referring to https://github.com/allenai/allennlp/blob/master/allennlp/models/crf_tagger.py#L229-L239
        if labels is not None:
            log_likelihood = self.crf(logits, labels, sentence_masks)
            output_dict["loss"] = -log_likelihood

            class_probabilities = logits * 0.0
            for i, instance_labels in enumerate(predicted_labels):
                for j, label_id in enumerate(instance_labels):
                    class_probabilities[i, j, label_id] = 1

            for metric in self.metrics.values():
                metric(class_probabilities, labels, sentence_masks)

        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Coverts tag ids to actual tags.
        """
        output_dict["labels"] = [
            [
                self.vocab.get_token_from_index(label, namespace="labels")
                for label in instance_labels
            ]
            for instance_labels in output_dict["labels"]
        ]
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "f1": self.metrics["f1"].get_metric(reset=reset)[2],
            "accuracy": self.metrics["accuracy"].get_metric(reset=reset),
        }

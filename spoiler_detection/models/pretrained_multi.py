from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import AutoConfig, AutoModel, AutoTokenizer

from spoiler_detection.metrics import get_accuracy
from spoiler_detection.models.base_model import BaseModel


class PretrainedMultiSentenceModel(BaseModel):
    def __init__(self, hparams):
        super(PretrainedMultiSentenceModel, self).__init__(hparams)

        self.config = AutoConfig.from_pretrained(hparams.model_type)
        self.model = AutoModel.from_config(self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.model_type)

        self._use_genres = hparams.use_genres
        if hparams.use_genres:
            classifier_input_dim = self.model.config.hidden_size + 10
        else:
            classifier_input_dim = self.model.config.hidden_size

        self.dropout = nn.Dropout(hparams.classifier_dropout_prob)
        self.classifier = nn.Linear(classifier_input_dim, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

        self.init_weights(self.classifier)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        genres=None,
        labels=None,
    ):
        pooled_outputs = []
        for (i, a, t) in zip(input_ids, attention_mask, token_type_ids):
            pooled_outputs.append(
                self.model(input_ids=i, attention_mask=a, token_type_ids=t)[1]
            )

        pooled_output = torch.stack(pooled_outputs)

        pooled_output = self.dropout(pooled_output)

        if self._use_genres:
            duplicated_genres = torch.stack([genres for _ in range(labels.shape[1])], 1)
            pooled_output = torch.cat((pooled_output, duplicated_genres), dim=-1)

        logits = self.classifier(pooled_output)

        if labels is not None:
            labels_mask = labels != -1
            log_likelihood = self.crf(logits, labels, mask=labels_mask)
            predicted_labels = self.crf.decode(logits, mask=labels_mask)

            probs = torch.tensor(
                [
                    [1 if i == label_id else 0 for i in range(self.num_labels)]
                    for instance_labels in predicted_labels
                    for label_id in instance_labels
                ]
            )
            loss = -log_likelihood
            return probs, loss, labels[labels != -1]

        return probs

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, genres, labels = batch
        probs, loss, flattened_labels = self(
            input_ids, attention_mask, token_type_ids, genres, labels
        )

        acc = get_accuracy(probs, flattened_labels)
        metrics = {
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            "train_acc": acc,
        }

        return {"loss": loss, "log": metrics, "train_acc": acc}

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, genres, labels = batch
        probs, loss, flattened_labels = self(
            input_ids, attention_mask, token_type_ids, genres, labels
        )

        return {"val_loss": loss, "probs": probs, "labels": flattened_labels}

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, genres, labels = batch
        probs, loss, flattened_labels = self(
            input_ids, attention_mask, token_type_ids, genres, labels
        )

        return {"test_loss": loss, "probs": probs, "labels": flattened_labels}

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = BaseModel.add_model_specific_args(parent_parser)
        parser.add_argument("--model_type", type=str, default="albert-base-v2")
        parser.add_argument("--use_genres", action="store_true")
        parser.add_argument("--classifier_dropout_prob", type=float, default=0.1)
        parser.add_argument("--positive_class_weight", type=float, default=0.5)
        return parser

from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from spoiler_detection.metrics import get_accuracy
from spoiler_detection.models.base_model import BaseModel


class PretrainedSingleSentenceModel(BaseModel):
    def __init__(self, dataset, hparams):
        super(PretrainedSingleSentenceModel, self).__init__(dataset, hparams)

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
        self.loss = torch.nn.CrossEntropyLoss(
            weight=torch.FloatTensor(
                [1 - hparams.positive_class_weight, hparams.positive_class_weight]
            )
        )

        self.init_weights(self.classifier)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        genres=None,
        labels=None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        if self._use_genres:
            pooled_output = torch.cat((pooled_output, genres), dim=-1)

        logits = self.classifier(pooled_output)
        probs = F.softmax(logits, dim=-1)

        if labels is not None:
            loss = self.loss(logits, labels)
            return probs, loss

        return probs

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, genres, labels = batch
        probs, loss = self(input_ids, attention_mask, token_type_ids, genres, labels)

        acc = get_accuracy(probs, labels)
        metrics = {
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            "train_acc": acc,
        }

        return {"loss": loss, "log": metrics, "train_acc": acc}

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, genres, labels = batch
        probs, loss = self(input_ids, attention_mask, token_type_ids, genres, labels)

        return {"val_loss": loss, "probs": probs, "labels": labels}

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, genres, labels = batch
        probs, loss = self(input_ids, attention_mask, token_type_ids, genres, labels)

        return {"test_loss": loss, "probs": probs, "labels": labels}

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = BaseModel.add_model_specific_args(parent_parser)
        parser.add_argument("--model_type", type=str, default="albert-base-v2")
        parser.add_argument("--use_genres", action="store_true")
        parser.add_argument("--classifier_dropout_prob", type=float, default=0.1)
        parser.add_argument("--positive_class_weight", type=float, default=0.5)
        return parser

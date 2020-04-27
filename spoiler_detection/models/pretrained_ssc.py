import logging
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from spoiler_detection.metrics import get_accuracy
from spoiler_detection.models.base_model import BaseModel


class PretrainedSscModel(BaseModel):
    def __init__(self, hparams):
        super(PretrainedSscModel, self).__init__(hparams)

        self.config = AutoConfig.from_pretrained(hparams.model_type)
        self.model = AutoModel.from_config(self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.model_type)
        self._sep_token_id = self.tokenizer._convert_token_to_id("[SEP]")

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

        last_hidden_states = outputs[0]

        last_hidden_states = self.dropout(last_hidden_states)

        sep_mask = input_ids == self._sep_token_id
        sep_embeddings = last_hidden_states[sep_mask]
        num_sentences = sep_embeddings.shape[0]

        if self._use_genres:
            flattened_genres = (genres[genres != -1]).reshape(num_sentences, 10)
            sep_embeddings = torch.cat((sep_embeddings, flattened_genres), dim=-1)

        logits = self.classifier(sep_embeddings)
        probs = F.softmax(logits, dim=-1)

        if labels is not None:
            flattened_labels = labels[labels != -1]
            
            loss = self.loss(logits, flattened_labels)
            return probs, loss, flattened_labels

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
        parser.add_argument("--use_genres", type=int, choices={0, 1}, default=0)
        parser.add_argument("--classifier_dropout_prob", type=float, default=0.1)
        parser.add_argument("--positive_class_weight", type=float, default=0.5)
        return parser

from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from spoiler_detection.metrics import get_training_metrics, get_validation_metrics
from spoiler_detection.models.base_model import BaseModel


class PretrainedSscModel(BaseModel):
    def __init__(self, dataset, hparams):
        super(PretrainedSscModel, self).__init__(dataset, hparams)

        self.config = AutoConfig.from_pretrained(hparams.model_type)
        self.model = AutoModel.from_config(self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.model_type)
        self._sep_token_id = self.tokenizer._convert_token_to_id("[SEP]")

        hparams.use_genres = True
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
            ),
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
            flattened_genres = (genres[genres != -1])[: num_sentences * 10].reshape(
                num_sentences, 10
            )
            sep_embeddings = torch.cat((sep_embeddings, flattened_genres), dim=-1)

        logits = self.classifier(sep_embeddings)

        if labels is not None:
            flattened_labels = labels[labels != -1]
            num_labels = flattened_labels.shape[0]

            if num_sentences < num_labels:
                print(f"Found {num_labels} labels but {num_sentences} sentences")
                flattened_labels = flattened_labels[:num_sentences]
            loss = self.loss(
                logits.view(-1, self.num_labels), flattened_labels.view(-1)
            )
            return logits, loss, flattened_labels

        return logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, genres, labels = batch
        logits, loss, flattened_labels = self(
            input_ids, attention_mask, token_type_ids, genres, labels
        )
        probs = F.softmax(logits, dim=-1)

        metrics = get_training_metrics(probs, flattened_labels)
        return {"loss": loss, "log": metrics, "progress_bar": metrics}

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, genres, labels = batch
        logits, loss, flattened_labels = self(
            input_ids, attention_mask, token_type_ids, genres, labels
        )
        probs = F.softmax(logits, dim=-1)

        return {"val_loss": loss, "probs": probs, "labels": flattened_labels}

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, genres, labels = batch
        logits, loss, flattened_labels = self(
            input_ids, attention_mask, token_type_ids, genres, labels
        )
        probs = F.softmax(logits, dim=-1)

        return {"test_loss": loss, "probs": probs, "labels": flattened_labels}

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = BaseModel.add_model_specific_args(parent_parser)
        parser.add_argument("--model_type", type=str, default="albert-base-v2")
        parser.add_argument("--use_genres", action="store_true")
        parser.add_argument("--classifier_dropout_prob", type=float, default=0.1)
        parser.add_argument("--positive_class_weight", type=float, default=0.5)
        return parser

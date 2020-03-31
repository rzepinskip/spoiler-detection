import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from spoiler_detection.metrics import get_training_metrics, get_validation_metrics
from spoiler_detection.models.base_model import BaseModel


class PretrainedSingleSentenceModel(BaseModel):
    def __init__(self, dataset, model_type, use_genres=False):
        super(PretrainedSingleSentenceModel, self).__init__(dataset)

        self.model = AutoModel.from_pretrained(model_type)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)

        self._use_genres = use_genres
        if use_genres:
            classifier_input_dim = self.model.config.hidden_size + 10
        else:
            classifier_input_dim = self.model.config.hidden_size

        self._classification_layer = nn.Linear(classifier_input_dim, 2)
        self._loss = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.8, 0.2]))

    def forward(self, input_ids, attention_mask, token_type_ids, genres):
        h, _ = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        h_cls = h[:, 0]

        if self._use_genres:
            h_cls = torch.cat((h_cls, genres), dim=-1)

        logits = self._classification_layer(h_cls)
        return logits

    def training_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, genres, label = batch

        logits = self(input_ids, attention_mask, token_type_ids, genres)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        loss = self._loss(logits, label)
        metrics = get_training_metrics(probs, label)
        return {"loss": loss, "log": metrics, "progress_bar": metrics}

    def validation_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, genres, label = batch

        logits = self(input_ids, attention_mask, token_type_ids, genres)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        loss = self._loss(logits, label)
        return {"loss": loss, "probs": probs, "label": label}

    def configure_optimizers(self):
        return torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08
        )

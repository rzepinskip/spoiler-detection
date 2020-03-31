import torch
import torch.nn as nn
from transformers import AutoTokenizer

from spoiler_detection.datasets.goodreads import GoodreadsSingleSentenceDataset
from spoiler_detection.metrics import get_training_metrics, get_validation_metrics
from spoiler_detection.models.base_model import BaseModel


class BasicModel(BaseModel):
    def __init__(self, dataset, hparams):
        super(BasicModel, self).__init__(dataset)

        self.tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        self.dataset = dataset

        self._classification_layer = nn.Linear(10, 2)
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, genres):
        logits = self._classification_layer(genres)
        return logits

    def training_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, genres, label = batch

        logits = self(input_ids, attention_mask, token_type_ids, genres)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        loss = self._loss(logits, label)
        metrics = get_training_metrics(probs, label)
        metrics["train_loss"] = loss
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

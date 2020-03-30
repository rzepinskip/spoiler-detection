import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score
from spoiler_detection.data_readers.goodreads import GoodreadsSingleSentenceDataset


class PretrainedSingleSentenceModel(pl.LightningModule):
    def __init__(self, model_type, dataset, use_genres=False):
        super(PretrainedSingleSentenceModel, self).__init__()

        self.model = AutoModel.from_pretrained(model_type)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.dataset = dataset

        self._use_genres = use_genres
        if use_genres:
            self.W = nn.Linear(self.model.config.hidden_size + 10, 2)
        else:
            self.W = nn.Linear(self.model.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids, genres):
        h, _ = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        h_cls = h[:, 0]

        if self._use_genres:
            h_cls = torch.cat((h_cls, genres), dim=-1)

        logits = self.W(h_cls)
        return logits

    def training_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, genres, label = batch

        y_hat = self(input_ids, attention_mask, token_type_ids, genres)  # calls forward

        loss = F.cross_entropy(y_hat, label)

        a, y_hat = torch.max(y_hat, dim=1)
        train_acc = accuracy_score(y_hat.cpu(), label.cpu())
        train_acc = torch.tensor(train_acc)
        tensorboard_logs = {"train_acc": train_acc, "train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, genres, label = batch

        y_hat = self(input_ids, attention_mask, token_type_ids, genres)

        loss = F.cross_entropy(y_hat, label)

        a, y_hat = torch.max(y_hat, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), label.cpu())
        val_acc = torch.tensor(val_acc)

        return {"val_loss": loss, "val_acc": val_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_val_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        tensorboard_logs = {"val_loss": avg_loss, "avg_val_acc": avg_val_acc}
        return {"avg_val_loss": avg_loss, "progress_bar": tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08
        )

    def prepare_data(self):
        self.train_dl = self.dataset.get_dataloader("train", self.tokenizer)
        self.val_dl = self.dataset.get_dataloader("val", self.tokenizer)

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

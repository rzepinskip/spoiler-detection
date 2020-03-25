import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score
from spoiler_detection.data_readers.goodreads import get_goodreads_dataset


class BertFinetuner(pl.LightningModule):
    def __init__(self):
        super(BertFinetuner, self).__init__()

        model_type = "bert-base-cased"
        self.bert = BertModel.from_pretrained(model_type, output_attentions=True)
        self.tokenizer = BertTokenizer.from_pretrained(model_type)

        self.W = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        h, _, attn = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        h_cls = h[:, 0]
        logits = self.W(h_cls)
        return logits, attn

    def training_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch

        y_hat, attn = self(input_ids, attention_mask, token_type_ids)

        loss = F.cross_entropy(y_hat, label)

        a, y_hat = torch.max(y_hat, dim=1)
        train_acc = accuracy_score(y_hat.cpu(), label.cpu())
        train_acc = torch.tensor(train_acc)
        tensorboard_logs = {"train_acc": train_acc, "train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch

        y_hat, attn = self(input_ids, attention_mask, token_type_ids)

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
        self.train_dl = get_goodreads_dataset(self.tokenizer, "train")
        self.val_dl = get_goodreads_dataset(self.tokenizer, "val")

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

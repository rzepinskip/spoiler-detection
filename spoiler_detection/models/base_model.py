from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

from spoiler_detection.metrics import get_test_metrics, get_validation_metrics


class BaseModel(pl.LightningModule):
    def __init__(self, hparams):
        super(BaseModel, self).__init__()
        self.hparams = hparams
        self.num_labels = 2

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["train_acc"] for x in outputs]).mean()

        metrics = {
            "epoch": self.current_epoch,
            "avg_train_loss": avg_loss,
            "avg_train_acc": avg_acc,
        }
        return {"log": metrics}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        probs = torch.cat([x["probs"] for x in outputs])
        label = torch.cat([x["labels"] for x in outputs])

        if self.global_step == 0:
            return {"val_loss": avg_loss}

        metrics = get_validation_metrics(probs, label)
        metrics["epoch"] = self.current_epoch
        metrics["avg_val_loss"] = avg_loss

        return {
            "val_loss": avg_loss,
            "log": metrics,
            "progress_bar": metrics,
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        probs = torch.cat([x["probs"] for x in outputs])
        label = torch.cat([x["labels"] for x in outputs])

        metrics = get_test_metrics(probs, label)

        return {
            "test_loss": avg_loss,
            "log": metrics,
            "progress_bar": metrics,
        }

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        if self.trainer is None:
            # learning rate finding mode
            return optimizer

        n_devices = 1
        if self.hparams.tpu_cores:
            n_devices = self.hparams.tpu_cores
        elif self.hparams.gpus:
            n_devices = max(1, self.hparams.gpus)
        t_total = (
            len(self.train_dl)
            * self.trainer.train_percent_check
            // n_devices
            // self.hparams.accumulate_grad_batches
            * float(self.hparams.epochs)
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0.1 * t_total, num_training_steps=t_total,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def prepare_data(self):
        self.train_dl = self.hparams.dataset.get_dataloader(
            "train", self.tokenizer, self.hparams.batch_size
        )
        self.val_dl = self.hparams.dataset.get_dataloader(
            "val", self.tokenizer, self.hparams.batch_size
        )
        self.test_dl = self.hparams.dataset.get_dataloader(
            "test", self.tokenizer, self.hparams.batch_size
        )

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            "--learning_rate",
            default=2e-5,
            type=float,
            help="The initial learning rate for Adam.",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Weight decay if we apply some.",
        )
        parser.add_argument(
            "--adam_epsilon",
            default=1e-6,
            type=float,
            help="Epsilon for Adam optimizer.",
        )
        parser.add_argument(
            "--epochs",
            default=4,
            type=int,
            help="Total number of training epochs to perform.",
        )
        parser.add_argument(
            "--accumulate_grad_batches",
            default=1,
            type=int,
            help="Accumulates grads every k batches.",
        )

        parser.add_argument("--batch_size", default=32, type=int)

        return parser

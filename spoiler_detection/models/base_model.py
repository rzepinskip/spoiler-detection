from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

from spoiler_detection.metrics import get_test_metrics, get_validation_metrics


class BaseModel(pl.LightningModule):
    def __init__(self, dataset):
        super(BaseModel, self).__init__()
        self.dataset = dataset

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        probs = torch.cat([x["probs"] for x in outputs])
        label = torch.cat([x["label"] for x in outputs])

        if self.global_step == 0:
            return {"val_loss": avg_loss}

        metrics = get_validation_metrics(probs, label)
        metrics["epoch"] = self.current_epoch

        return {
            "val_loss": avg_loss,
            "log": metrics,
            "progress_bar": metrics,
        }

    def test_epoch_end(self, outputs):
        probs = torch.cat([x["probs"] for x in outputs])
        label = torch.cat([x["label"] for x in outputs])

        metrics = get_test_metrics(probs, label)

        return {
            "log": metrics,
            "progress_bar": metrics,
        }

    def prepare_data(self):
        self.train_dl = self.dataset.get_dataloader("train", self.tokenizer)
        self.val_dl = self.dataset.get_dataloader("val", self.tokenizer)
        self.test_dl = self.dataset.get_dataloader("test", self.tokenizer)

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser

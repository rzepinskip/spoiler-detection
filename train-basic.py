import wandb
from pytorch_lightning import Trainer

from spoiler_detection.datasets import GoodreadsSingleSentenceDataset
from spoiler_detection.loggers import ResumableTestTubeLogger, ResumableWandbLogger
from spoiler_detection.models import BasicModel

if __name__ == "__main__":
    dataset = GoodreadsSingleSentenceDataset(max_length=128)
    model = BasicModel(dataset)

    wandb_logger = ResumableWandbLogger()
    trainer = Trainer(
        logger=wandb_logger,
        progress_bar_refresh_rate=100,
        train_percent_check=0.05,
        val_percent_check=0.05,
        max_epochs=4,
        default_save_path=wandb_logger.get_checkpoints_root(),
        resume_from_checkpoint=wandb_logger.get_last_checkpoint(),
    )
    trainer.fit(model)

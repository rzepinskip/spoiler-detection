from pytorch_lightning import Trainer

from spoiler_detection.datasets import GoodreadsSingleSentenceDataset
from spoiler_detection.loggers import ResumableTestTubeLogger
from spoiler_detection.models import BasicModel

if __name__ == "__main__":
    dataset = GoodreadsSingleSentenceDataset(max_length=128)
    model = BasicModel(dataset)

    logger = ResumableTestTubeLogger()
    trainer = Trainer(
        logger=logger,
        progress_bar_refresh_rate=100,
        train_percent_check=0.05,
        val_percent_check=0.05,
        resume_from_checkpoint=logger.get_last_checkpoint(),
        max_epochs=2,
    )
    trainer.fit(model)

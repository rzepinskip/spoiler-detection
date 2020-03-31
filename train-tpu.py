from pytorch_lightning import Trainer

from spoiler_detection.datasets import GoodreadsSingleSentenceDataset
from spoiler_detection.loggers import ResumableTestTubeLogger
from spoiler_detection.models import PretrainedSingleSentenceModel

if __name__ == "__main__":
    dataset = GoodreadsSingleSentenceDataset(max_length=128)
    model = PretrainedSingleSentenceModel(dataset, "albert-base-v2")

    logger = ResumableTestTubeLogger()
    trainer = Trainer(
        logger=logger,
        progress_bar_refresh_rate=10,
        max_epochs=4,
        num_tpu_cores=8,
        resume_from_checkpoint=logger.get_last_checkpoint(),
    )
    trainer.fit(model)

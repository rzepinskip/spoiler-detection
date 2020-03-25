from spoiler_detection.models.basic_bert import BertFinetuner
from pytorch_lightning import Trainer
from spoiler_detection.loggers import ResumableTestTubeLogger

if __name__ == "__main__":
    model = BertFinetuner()

    logger = ResumableTestTubeLogger()
    trainer = Trainer(
        logger=logger,
        progress_bar_refresh_rate=10,
        max_epochs=4,
        num_tpu_cores=8,
        resume_from_checkpoint=logger.get_last_checkpoint(),
    )
    trainer.fit(model)

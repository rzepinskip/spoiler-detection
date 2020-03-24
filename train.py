from spoiler_detection.models.basic_bert import BertFinetuner
from pytorch_lightning import Trainer

if __name__ == "__main__":
    model = BertFinetuner()

    trainer = Trainer(progress_bar_refresh_rate=1, max_epochs=2, fast_dev_run=True)
    trainer.fit(model)

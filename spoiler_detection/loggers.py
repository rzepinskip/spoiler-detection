from pytorch_lightning.loggers import TestTubeLogger
import os
import glob


class ResumableTestTubeLogger(TestTubeLogger):
    def __init__(self):
        super(ResumableTestTubeLogger, self).__init__(save_dir="lightning_logs")

    def get_last_checkpoint(self):
        if self._experiment is None:
            return None

        last_version_checkpoints_folder = os.path.join(
            self.save_dir,
            self._name,
            f"version_{self._experiment._Experiment__get_last_experiment_version()}",
            "checkpoints",
        )

        return sorted(
            glob.glob(os.path.join(last_version_checkpoints_folder, "*.ckpt",))
        )[-1]

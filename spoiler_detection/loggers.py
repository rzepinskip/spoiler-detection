import glob
import os

import wandb
from pytorch_lightning.loggers import TestTubeLogger, WandbLogger


class ResumableTestTubeLogger(TestTubeLogger):
    def __init__(self, version=None):
        if version is None:
            version = max(self.__get_last_experiment_version(), 0)
        super(ResumableTestTubeLogger, self).__init__(
            save_dir="lightning_logs", version=version
        )

    def get_last_checkpoint(self):
        if self._experiment is None:
            return None

        last_version_checkpoints_folder = os.path.join(
            "lightning_logs",
            "default",
            f"version_{self.__get_last_experiment_version()}",
            "checkpoints",
        )

        return sorted(
            glob.glob(os.path.join(last_version_checkpoints_folder, "*.ckpt",))
        )[-1]

    def __get_last_experiment_version(self):
        try:
            exp_cache_file = os.path.join("lightning_logs", "default")
            last_version = -1
            for f in os.listdir(exp_cache_file):
                if "version_" in f:
                    file_parts = f.split("_")
                    version = int(file_parts[-1])
                    last_version = max(last_version, version)
            return last_version
        except Exception as e:
            return -1


class ResumableWandbLogger(WandbLogger):
    def __init__(self, id=None, offline=False):
        super().__init__(id=id, offline=offline, project="spoiler_detection")
        if id is not None and offline == False:
            run = wandb.Api().run(f"rzepinskip/spoiler_detection/{id}")
            for file in run.files():
                if file.name.endswith(".ckpt"):
                    print(f"Downloading {file.name}")
                    file.download(root=self.experiment.dir)

    def log_hyperparams(self, params) -> None:
        params = self._convert_params(params)
        self.experiment.config.update(params, allow_val_change=True)

    def get_checkpoints_root(self):
        return self.experiment.dir

    def get_last_checkpoint(self):
        if self.experiment.resumed == False:
            return None

        last_version_checkpoints_folder = os.path.join(
            self.experiment.dir,
            "spoiler_detection",
            f"version_{self.experiment.id}",
            "checkpoints",
        )

        checkpoints = sorted(
            glob.glob(os.path.join(last_version_checkpoints_folder, "*.ckpt",))
        )

        if len(checkpoints) == 0:
            raise ValueError("No checkpoints to restore")

        return checkpoints[-1]

from argparse import ArgumentParser

from pytorch_lightning import Trainer

from spoiler_detection.datasets import (
    GoodreadsMultiSentenceDataset,
    GoodreadsSingleSentenceDataset,
    GoodreadsSscDataset,
    TvTropesMovieSingleSentenceDataset,
)
from spoiler_detection.loggers import ResumableWandbLogger
from spoiler_detection.models import PretrainedSingleSentenceModel, PretrainedSscModel

MODELS = {
    "PretrainedSingleSentenceModel": PretrainedSingleSentenceModel,
    "PretrainedSscModel": PretrainedSscModel,
}

DATASETS = {
    "GoodreadsMultiSentenceDataset": GoodreadsMultiSentenceDataset,
    "GoodreadsSingleSentenceDataset": GoodreadsSingleSentenceDataset,
    "GoodreadsSscDataset": GoodreadsSscDataset,
    "TvTropesMovieSingleSentenceDataset": TvTropesMovieSingleSentenceDataset,
}


def main(args):
    dataset = DATASETS[args.dataset_name](hparams=args)
    model = MODELS[args.model_name](dataset=dataset, hparams=args)

    wandb_logger = ResumableWandbLogger(id=args.run_id, offline=args.dry_run)
    wandb_logger.log_hyperparams(args)

    params = {
        "logger": wandb_logger,
        "default_save_path": wandb_logger.get_checkpoints_root(),
        "resume_from_checkpoint": wandb_logger.get_last_checkpoint(),
    }

    if args.dry_run:
        params["fast_dev_run"] = True

    if args.tpu:
        params["num_tpu_cores"] = 8
        params["precision"] = 16

    trainer = Trainer(**params)
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    # Choose model and dataset
    parser.add_argument(
        "--model_name",
        type=str,
        default="PretrainedSingleSentenceModel",
        help=str(DATASETS.keys()),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="TvTropesMovieSingleSentenceDataset",
        help=str(MODELS.keys()),
    )
    parser.add_argument(
        "--run_id", type=str, help="Id of Wandb session to resume",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--tpu", action="store_true")
    temp_args, _ = parser.parse_known_args()

    parser = MODELS[temp_args.model_name].add_model_specific_args(parser)
    parser = DATASETS[temp_args.dataset_name].add_dataset_specific_args(parser)
    parser = ArgumentParser(parents=[parser])

    args = parser.parse_args()

    main(args)

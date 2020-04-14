from argparse import ArgumentParser
from multiprocessing import freeze_support

from pytorch_lightning import Trainer

from spoiler_detection.datasets import (
    GoodreadsMultiSentenceDataset,
    GoodreadsSingleSentenceDataset,
    GoodreadsSscDataset,
    TvTropesMovieSingleSentenceDataset,
)
from spoiler_detection.loggers import ResumableWandbLogger
from spoiler_detection.models import (
    PretrainedMultiSentenceModel,
    PretrainedSingleSentenceModel,
    PretrainedSscModel,
)

MODELS = {
    "PretrainedSingleSentenceModel": PretrainedSingleSentenceModel,
    "PretrainedMultiSentenceModel": PretrainedMultiSentenceModel,
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
    args.dataset = dataset

    model = MODELS[args.model_name](hparams=args)

    wandb_logger = ResumableWandbLogger(id=args.run_id, offline=True)
    wandb_logger.log_hyperparams(args)

    params = {
        "logger": wandb_logger,
        "default_save_path": wandb_logger.get_checkpoints_root(),
        "resume_from_checkpoint": wandb_logger.get_last_checkpoint(),
        "max_epochs": args.epochs,
        "train_percent_check": args.dataset_percent,
        "val_percent_check": args.dataset_percent,
    }

    if args.gpus:
        params["gpus"] = args.gpus

    trainer = Trainer(**params)
    # Run learning rate finder
    lr_finder = trainer.lr_find(model)

    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.show()


if __name__ == "__main__":
    freeze_support()
    parser = ArgumentParser(add_help=False)

    parser.add_argument(
        "--model_name",
        type=str,
        default="PretrainedSingleSentenceModel",
        help=str(MODELS.keys()),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="TvTropesMovieSingleSentenceDataset",
        help=str(DATASETS.keys()),
    )
    parser.add_argument(
        "--run_id", type=str, help="Id of Wandb session to resume",
    )
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--dataset_percent", type=float, default=1.0)
    temp_args, _ = parser.parse_known_args()

    parser = MODELS[temp_args.model_name].add_model_specific_args(parser)
    parser = DATASETS[temp_args.dataset_name].add_dataset_specific_args(parser)
    parser = ArgumentParser(parents=[parser])

    args = parser.parse_args()

    main(args)

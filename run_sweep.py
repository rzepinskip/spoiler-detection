import wandb

sweep_config = {
    "name": "Test sweep 2",
    "program": "train.py",
    "method": "grid",
    "parameters": {
        "model_name": {"values": ["PretrainedSingleSentenceModel"]},
        "dataset_name": {"values": ["GoodreadsSingleSentenceDataset"]},
        "model_type": {"values": ["albert-base-v2"]},
        "classifier_dropout_prob": {"values": [0.1]},
        "epochs": {"values": [6]},
        "batch_size": {"values": [32]},
        "num_workers": {"values": [2]},
        "learning_rate": {"values": [2e-5, 4e-5]},
        "tpu_cores": {"values": [8]},
    },
    "metric": "avg_val_loss",
}

sweep_id = wandb.sweep(sweep_config, entity="rzepinskip", project="spoiler_detection")
wandb.agent(sweep_id)

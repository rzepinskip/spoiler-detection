{
  "dataset_reader": {
    "type": "goodreads_multiple_sentences"
  },
  "train_data_path": "tests/fixtures/goodreads.jsonl",
  "validation_data_path": "tests/fixtures/goodreads.jsonl",
  "test_data_path": "tests/fixtures/goodreads.jsonl",
  "evaluate_on_test": true,
  "model": {
    "type": "multiple_sentences_classifier-old",
    "class_weights": [0.05, 0.95],
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
        }
      }
    },
    "sentence_encoder": {
      "type": "boe",
      "embedding_dim": 2,
      "averaged": true
    }
  },
  "data_loader": {
    "num_workers": 0,
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 32
    }
  },
  "trainer": {
    "num_epochs": 1,
    "cuda_device": -1,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad"
    }
  }
}

local transformer_model = "albert-base-v2";
local transformer_dim = 768;
local cls_is_last_token = false;
local use_genres = true;
local max_length = 512;

{
  "dataset_reader":{
    "type": "goodreads_multiple_sentences-ssc",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model,
      "max_length": max_length,
      "add_special_tokens": false
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": max_length
      }
    }
  },
  "train_data_path": "tests/fixtures/goodreads.jsonl",
  "validation_data_path": "tests/fixtures/goodreads.jsonl",
  "model": {
    "type": "bert-ssc",
    "class_weights": [0.2, 0.8],
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
        }
      }
    },
    "dropout": 0.1
  },
  "data_loader": {
    "batch_size" : 1
  },
  "trainer": {
    "num_epochs": 1,
    "cuda_device" : -1,
    "validation_metric": "+accuracy",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-5,
      "weight_decay": 0.1,
    }
  }
}

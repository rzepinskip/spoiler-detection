local transformer_model = "albert-base-v2";
local transformer_dim = 768;
local cls_is_last_token = false;

{
  "dataset_reader":{
    "type": "goodreads_multiple_sentences",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
      }
    }
  },
  "train_data_path": "tests/fixtures/goodreads.jsonl",
  "validation_data_path": "tests/fixtures/goodreads.jsonl",
  "model": {
    "type": "multiple_sentences_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
        }
      }
    },
    "seq2vec_encoder": {
       "type": "cls_pooler",
       "embedding_dim": transformer_dim,
       "cls_is_last_token": cls_is_last_token
    },
    "feedforward": {
      "input_dim": transformer_dim,
      "num_layers": 1,
      "hidden_dims": transformer_dim,
      "activations": "tanh"
    },
    "dropout": 0.1
  },
  "data_loader": {
    "num_workers": 2,
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

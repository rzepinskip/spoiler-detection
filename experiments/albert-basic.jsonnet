local transformer_model = "albert-base-v2";
local transformer_dim = 768;
local cls_is_last_token = false;
local use_genres = false;
local max_length = 128;

{
  "dataset_reader":{
    "type": "goodreads_single_sentence",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model,
      "max_length": max_length
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": max_length
      }
    }
  },
  "train_data_path": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-val.json.gz",
  "validation_data_path": "https://spoiler-datasets.s3.eu-central-1.amazonaws.com/goodreads_balanced-test.json.gz",
  "model": {
    "type": "single_sentence_classifier",
    "class_weights": [0.2, 0.8],
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
    "use_genres": use_genres,
    "feedforward": {
      "input_dim": if use_genres then transformer_dim+10 else transformer_dim,
      "num_layers": 1,
      "hidden_dims": if use_genres then transformer_dim+10 else transformer_dim,
      "activations": "tanh"
    },
    "dropout": 0.1
  },
  "data_loader": {
    "batch_size" : 32
  },
  "trainer": {
    "num_epochs": 1,
    "cuda_device" : 0,
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

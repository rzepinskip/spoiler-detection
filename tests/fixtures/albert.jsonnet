local transformer_model = "albert-base-v2";
local transformer_dim = 768;
local cls_is_last_token = false;
local use_genres = true;

{
  "dataset_reader":{
    "type": "goodreads_single_sentence",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model,
      "add_special_tokens": true
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
    "type": "single_sentence_classifier",
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

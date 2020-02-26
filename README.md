```
allennlp train tests/fixtures/single_sentence_classifier.json \
    -s tmp \
    --include-package spoiler_detection
allennlp predict tests/fixtures/single_sentence_classifier.tar.gz \
    tests/fixtures/goodreads.jsonl \
    --include-package spoiler_detection \
    --predictor single_sentence_predictor \
    --use-dataset-reader
allennlp predict tests/fixtures/single_sentence_classifier.tar.gz \
    tests/fixtures/goodreads.jsonl \
    --include-package spoiler_detection \
    --use-dataset-reader
```

```
allennlp train tests/fixtures/multiple_sentences_classifier.json \
    -s tmp \
    --include-package spoiler_detection
```

```
allennlp train experiments/lstm_sentence_encoder_with_glove.json \
    -s tmp \
    --include-package spoiler_detection \
    --file-friendly-logging
```

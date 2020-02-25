```
allennlp train tests/fixtures/single_sentence_classifier.json \
    -s output \
    --include-package spoiler_detection
allennlp predict tests/fixtures/single_sentence_classifier.tar.gz \
    tests/fixtures/goodreads.jsonl \
    --include-package spoiler_detection \
    --predictor single_sentence_predictor \
    --use-dataset-reader
```

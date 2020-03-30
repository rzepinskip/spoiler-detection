```
allennlp train tests/fixtures/albert.jsonnet \
    -s tmp \
    --include-package spoiler_detection \
    --file-friendly-logging \
    -f
```

```
allennlp train tests/fixtures/albert-crf.jsonnet \
    -s tmp \
    --include-package spoiler_detection \
    --file-friendly-logging \
    -f
```

```
allennlp train tests/fixtures/albert-ssc.jsonnet \
    -s tmp \
    --include-package spoiler_detection \
    --file-friendly-logging \
    -f
```

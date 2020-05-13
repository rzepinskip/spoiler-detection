```
python train.py --model_name PooledModel --dataset_name GoodreadsSingleDataset --batch_size 2 --dry_run 1
python train.py --model_name PooledModel --dataset_name TvTropesMovieSingleDataset --batch_size 2 --dry_run 1
```

```
python test.py --model_name PooledModel --dataset_name TvTropesMovieSingleDataset --batch_size 2 --checkpoint "./checkpoint.tf"
```

```
python train.py --model_name PooledModel --dataset_name GoodreadsSingleDataset --model_type bert-base-uncased --batch_size 2 --dry_run 1
python train.py --model_name SequenceModel --dataset_name GoodreadsSingleDataset --model_type bert-base-uncased --batch_size 2 --dry_run 1
```

```
python train.py --model_name PooledModel --dataset_name GoodreadsSingleDataset --model_type bert-base-cased --batch_size 2 --dry_run 1
python train.py --model_name PooledModel --dataset_name GoodreadsSingleDataset --model_type albert-xlarge-v2 --batch_size 2 --dry_run 1
python train.py --model_name PooledModel --dataset_name GoodreadsSingleDataset --model_type roberta-base --batch_size 2 --dry_run 1
```

```
python train.py --model_name PooledModel --dataset_name GoodreadsSingleGenreAppendedDataset --model_type bert-base-uncased --batch_size 2 --dry_run 1
```

```
python train.py --model_name SequenceModel --dataset_name GoodreadsSingleDataset --model_type bert-base-uncased --batch_size 2 --dry_run 1 --loss focal
```


```
python train.py --model_name PooledModel --dataset_name GoodreadsSingleDataset --batch_size 2 --dry_run 1
python train.py --model_name PooledModel --dataset_name TvTropesMovieSingleDataset --batch_size 2 --dry_run 1
```

```
python test.py --model_name PooledModel --dataset_name TvTropesMovieSingleDataset --batch_size 2 --checkpoint "./checkpoint.tf"
```

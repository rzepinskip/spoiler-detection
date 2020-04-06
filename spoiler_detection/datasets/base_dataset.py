from abc import ABC, abstractmethod
from argparse import ArgumentParser

from torch.utils.data import Dataset


class ListDataset(Dataset):
    def __init__(self, data):
        super(ListDataset).__init__()
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class BaseDataset:
    @abstractmethod
    def get_dataloader(self, dataset_type, tokenizer, batch_size):
        pass

    @classmethod
    def add_dataset_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--max_length", type=int, default=128)
        return parser

from abc import ABC, abstractmethod

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
    def get_dataloader(self, dataset_type, tokenizer):
        pass

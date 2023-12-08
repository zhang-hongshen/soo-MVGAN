import pandas as pd
from torch.utils.data import Dataset
from typing import List


class MyDataset(Dataset):
    def __init__(self, filepath: str, columns: List[str] = None):
        self.data = pd.read_csv(filepath, usecols=columns)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.loc[idx].tolist()

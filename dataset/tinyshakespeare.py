import torch
from torch.utils.data import Dataset

class TinyShakespeareDataset(Dataset):
    def __init__(self, path: str, tokenizer, context_length: int):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        ids = tokenizer.encode(text)
        self.data = torch.tensor(ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        x = self.data[idx : idx+self.context_length]
        y = self.data[idx+1 : idx+1+self.context_length]
        return x, y

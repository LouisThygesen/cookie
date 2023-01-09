import numpy as np
import torch
from torch.utils.data import Dataset

class mnist(Dataset):
    def __init__(self, path):
        # Load already preprocessed content
        self.imgs = torch.load(f"{path}/images.pt")
        self.labels = torch.load(f"{path}/labels.pt")

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        imgs, labels = (self.imgs[idx], self.labels[idx])

        return imgs.float(), labels

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class mnist(Dataset):
    def __init__(self, file):
        # Load already preprocessed content 
        self.imgs = torch.tensor(np.load(file)['images'])
        self.labels = torch.tensor(np.load(file)['labels'])
        
    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        imgs, labels = (self.imgs[idx], self.labels[idx])

        return imgs.float(), labels

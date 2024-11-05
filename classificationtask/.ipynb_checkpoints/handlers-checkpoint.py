import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch

class EuroSAT_Handler(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)


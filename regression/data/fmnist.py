import argparse

import torch
import torchvision.datasets as tvds

from utils.paths import datasets_path
from utils.misc import gen_load_func

class FMNIST(tvds.FashionMNIST):
    def __init__(self, train=True, device='cpu'):
        super().__init__(datasets_path, train=train, download=True)
        self.data = self.data.unsqueeze(1).float().div(255).to(device)
        self.targets = self.targets.to(device)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

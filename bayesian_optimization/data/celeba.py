import torch
import os.path as osp
import argparse

from utils.paths import datasets_path
from utils.misc import gen_load_func

class CelebA(object):
    def __init__(self, train=True, size=32):
        self.data, self.targets = torch.load(
                osp.join(datasets_path, 'celeba',
                    f'train_{size}.pt' if train else f'eval_{size}.pt'))
        self.data = self.data.float() / 255.0

        if train:
            self.data, self.targets = self.data, self.targets
        else:
            self.data, self.targets = self.data, self.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

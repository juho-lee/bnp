import argparse

import torch
import torchvision.datasets as tvds

from utils.misc import gen_load_func

ROOT = '/nfs/parker/ext01/john/datasets'

class MNIST(tvds.MNIST):
    def __init__(self, *args, **kwargs):
        classes = kwargs.pop('classes', None)
        device = kwargs.pop('device', 'cuda')
        super().__init__(*args, **kwargs)

        self.data = self.data.unsqueeze(1).float().div(255).to(device)
        self.targets = self.targets.to(device)
        if classes is not None:
            idxs = []
            for c in classes:
                idxs.append(torch.where(self.targets==c)[0])
            idxs = torch.cat(idxs)
            self.data = self.data[idxs]
            self.targets = self.targets[idxs]

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def load_dataset(**kwargs):
    return MNIST(ROOT, train=True, **kwargs), MNIST(ROOT, train=False, **kwargs)

parser = argparse.ArgumentParser()
parser.add_argument('--classes', type=int, nargs='*', default=[0, 1, 2, 3, 4])
load = gen_load_func(parser, load_dataset)

import argparse
from attrdict import AttrDict

from utils.misc import gen_load_func

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torchmeta.utils.prototype import get_prototypes, prototypical_loss

from models.backbone import CNNBackBone

def accuracy(prototypes, fx, y):
    with torch.no_grad():
        sq_distances = (prototypes.unsqueeze(-3) - fx.unsqueeze(-2)).pow(2).sum(-1)
        return (sq_distances.argmin(-1) == y).float().mean()

class Proto(nn.Module):
    def __init__(self, in_channels, num_classes, hid_channels=64):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = CNNBackBone(in_channels, hid_channels=hid_channels)

    def forward(self, batch, num_samples=None):
        outs = AttrDict()
        xc, yc = batch['train'][0].cuda(), batch['train'][1].cuda()
        xt, yt = batch['test'][0].cuda(), batch['test'][1].cuda()

        fxc = self.backbone(xc)
        fxt = self.backbone(xt)

        prototypes = get_prototypes(fxc, yc, self.num_classes)
        outs.loss = prototypical_loss(prototypes, fxt, yt)

        with torch.no_grad():
            dist = (prototypes.unsqueeze(-3) - fxt.unsqueeze(-2)).pow(2).sum(-1)
            outs.acc = (dist.argmin(-1) == yt).float().mean()

        return outs

def load(args, cmdline):
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid_channels', type=int, default=64)
    sub_args, cmdline = parser.parse_known_args(cmdline)

    for k, v in sub_args.__dict__.items():
        args.__dict__[k] = v

    return Proto(3, args.ways,
            hid_channels=args.hid_channels), cmdline

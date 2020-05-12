import argparse
from attrdict import AttrDict

from utils.misc import gen_load_func

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from models.backbone import CNNBackBone
from models.attention import SelfAttn, MultiHeadAttn
from utils.misc import inv_centered_softmax, pad_logits

class CANP(nn.Module):
    def __init__(self, in_channels, num_classes,
            hid_channels=64, dim_hid=128):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = CNNBackBone(in_channels, hid_channels=hid_channels)
        self.self_attn = SelfAttn(hid_channels+num_classes-1, hid_channels)
        self.cross_attn = MultiHeadAttn(hid_channels, hid_channels, hid_channels,
                dim_hid, skip_con=False)
        self.fc = nn.Linear(dim_hid, num_classes-1)

    def predict(self, xc, yc, xt, num_samples=None, mask=None):
        fxc = self.backbone(xc)
        fxt = self.backbone(xt)

        yc_logits = inv_centered_softmax(
                F.one_hot(yc, self.num_classes).float())
        value = torch.cat([fxc, yc_logits], -1)
        value = self.self_attn(value, mask=mask)
        out = self.cross_attn(fxt, fxc, value, mask=mask)
        return Categorical(logits=pad_logits(self.fc(out)))

    def forward(self, batch, num_samples=None):
        outs = AttrDict()

        xc, yc = batch['train'][0].cuda(), batch['train'][1].cuda()
        xt, yt = batch['test'][0].cuda(), batch['test'][1].cuda()
        py = self.predict(xc, yc, xt)
        outs.ll = py.log_prob(yt).mean()
        outs.loss = -outs.ll
        outs.acc = (py.logits.argmax(-1) == yt).float().mean()
        return outs

def load(args, cmdline):
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid_channels', type=int, default=64)
    parser.add_argument('--dim_hid', type=int, default=128)
    sub_args, cmdline = parser.parse_known_args(cmdline)

    for k, v in sub_args.__dict__.items():
        args.__dict__[k] = v

    return CANP(3, args.ways,
            hid_channels=args.hid_channels, dim_hid=args.dim_hid), cmdline

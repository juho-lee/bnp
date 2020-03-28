import argparse

import torch
import torch.nn as nn
from attrdict import AttrDict

from utils.misc import add_args

from models.modules import AttEncoder, Decoder

class CANP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, fixed_var=False):
        super().__init__()
        self.enc = AttEncoder(dim_x=dim_x, dim_y=dim_y, dim_hid=dim_hid)
        self.dec = Decoder(dim_x=dim_x, dim_y=dim_y, dim_enc=dim_hid,
                dim_hid=dim_hid, fixed_var=fixed_var)

    def predict(self, xc, yc, xt, num_samples=None):
        return self.dec(self.enc(xc, yc, xt), xt)

    def forward(self, batch, num_samples=None, r_bs=0.0):
        outs = AttrDict()
        if self.training:
            py = self.predict(batch.xc, batch.yc, batch.x)
            outs.ll = py.log_prob(batch.y).sum(-1).mean()
            outs.loss = -outs.ll
        else:
            py = self.predict(batch.xc, batch.yc, batch.x)
            ll = py.log_prob(batch.y).sum(-1)
            num_ctx = batch.xc.shape[-2]
            outs.ctx_ll = ll[...,:num_ctx].mean()
            outs.tar_ll = ll[...,num_ctx:].mean()
        return outs

def load(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_hid', type=int, default=128)
    parser.add_argument('--fixed_var', '-fv', action='store_true', default=False)
    sub_args, _ = parser.parse_known_args()
    add_args(args, sub_args)
    return CANP(dim_hid=args.dim_hid, fixed_var=args.fixed_var)

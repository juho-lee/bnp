import argparse
import math

import torch
import torch.nn as nn
from attrdict import AttrDict

from utils.sampling import sample_with_partial_replacement
from utils.misc import add_args

from models.modules import Encoder, Decoder

class BNP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, fixed_var=False, r_bs=1.0):
        super().__init__()
        self.r_bs = r_bs
        self.denc = Encoder(dim_x=dim_x, dim_y=dim_y, dim_hid=dim_hid)
        self.benc = Encoder(dim_x=dim_x, dim_y=dim_y, dim_hid=dim_hid)
        self.dec = Decoder(dim_x=dim_x, dim_y=dim_y, dim_enc=2*dim_hid,
                dim_hid=dim_hid, fixed_var=fixed_var)

    def predict(self, xc, yc, xt, num_samples=None):
        K = num_samples or 1
        hid = self.denc(xc, yc)
        bxc, byc = sample_with_partial_replacement(
                [xc, yc], r=self.r_bs, num_samples=K)
        z = self.benc(bxc, byc)
        if K > 1:
            hid = torch.stack([hid]*K)
            xt = torch.stack([xt]*K)
        return self.dec(torch.cat([hid, z], -1), xt)

    def forward(self, batch, num_samples=None):
        outs = AttrDict()
        if self.training:
            py = self.predict(batch.xc, batch.yc, batch.x)
            outs.ll = py.log_prob(batch.y).sum(-1).mean()
            outs.loss = -outs.ll
        else:
            K = num_samples or 1
            py = self.predict(batch.xc, batch.yc, batch.x, num_samples=K)
            y = torch.stack([batch.y]*K)
            ll = py.log_prob(y).sum(-1).logsumexp(0) - math.log(K)
            num_ctx = batch.xc.shape[-2]
            outs.ctx_ll = ll[...,:num_ctx].mean()
            outs.tar_ll = ll[...,num_ctx:].mean()
        return outs

def load(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_hid', type=int, default=128)
    parser.add_argument('--r_bs', type=float, default=1.0)
    parser.add_argument('--fixed_var', '-fv', action='store_true', default=False)
    sub_args, _ = parser.parse_known_args()
    add_args(args, sub_args)
    return BNP(dim_hid=args.dim_hid, r_bs=args.r_bs, fixed_var=args.fixed_var)

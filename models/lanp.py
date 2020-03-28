import argparse
import math

import torch
import torch.nn as nn
from torch.distributions import kl_divergence, Normal
from attrdict import AttrDict

from utils.misc import add_args

from models.modules import AttEncoder, Encoder, Decoder

class LANP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128,
            dim_lat=128, fixed_var=False):
        super().__init__()
        self.denc = AttEncoder(dim_x=dim_x, dim_y=dim_y, dim_hid=dim_hid)
        self.lenc = AttEncoder(dim_x=dim_x, dim_y=dim_y,
                dim_hid=dim_hid, dim_lat=dim_lat)
        self.dec = Decoder(dim_x=dim_x, dim_y=dim_y, dim_enc=dim_hid+dim_lat,
                dim_hid=dim_hid, fixed_var=fixed_var)

    def predict(self, xc, yc, xt, num_samples=None):
        K = num_samples or 1
        hid = self.denc(xc, yc, xt)
        prior = self.lenc(xc, yc, xt)
        z = prior.rsample([K])
        hid = torch.stack([hid]*K)
        xt = torch.stack([xt]*K)
        encoded = torch.cat([hid, z], -1)
        return self.dec(encoded, xt)

    def forward(self, batch, num_samples=None):
        outs = AttrDict()
        if self.training:
            hid = self.denc(batch.xc, batch.yc, batch.x)
            prior = self.lenc(batch.xc, batch.yc, batch.x)
            posterior = self.lenc(batch.x, batch.y, batch.x)
            z = posterior.rsample()
            encoded = torch.cat([hid, z], -1)
            py = self.dec(encoded, batch.x)
            outs.recon = py.log_prob(batch.y).sum(-1).mean()
            outs.kld = kl_divergence(posterior, prior).sum(-1).mean()
            outs.loss = outs.kld  - outs.recon
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
    parser.add_argument('--dim_lat', type=int, default=128)
    parser.add_argument('--fixed_var', '-fv', action='store_true', default=False)
    sub_args, _ = parser.parse_known_args()
    add_args(args, sub_args)
    return LANP(dim_hid=args.dim_hid, dim_lat=args.dim_lat,
            fixed_var=args.fixed_var)

import argparse
import math

import torch
import torch.nn as nn
from torch.distributions import kl_divergence, Normal
from attrdict import AttrDict

from utils.misc import gen_load_func, logmeanexp

from models.modules import AttEncoder, Encoder, Decoder

class LANP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128):
        super().__init__()
        self.denc = AttEncoder(dim_x=dim_x, dim_y=dim_y, dim_hid=dim_hid)
        self.lenc = AttEncoder(dim_x=dim_x, dim_y=dim_y,
                dim_hid=dim_hid, dim_lat=dim_lat)
        self.dec = Decoder(dim_x=dim_x, dim_y=dim_y,
                dim_enc=dim_hid+dim_lat, dim_hid=dim_hid)

    def predict(self, xc, yc, xt, pz=None, num_samples=None):
        hid = self.denc(xc, yc, xt)
        pz = self.lenc(xc, yc, xt) if pz is None else pz
        if num_samples is None:
            z = pz.rsample()
        else:
            hid = torch.stack([hid]*num_samples)
            z = pz.rsample([num_samples])
            xt = torch.stack([xt]*num_samples)
        return self.dec(torch.cat([hid, z], -1), xt)

    def forward(self, batch, num_samples=None):
        outs = AttrDict()
        if self.training:
            prior = self.lenc(batch.xc, batch.yc, batch.x)
            posterior = self.lenc(batch.x, batch.y, batch.x)
            py = self.predict(batch.xc, batch.yc, batch.x, pz=posterior)
            outs.recon = py.log_prob(batch.y).sum(-1).mean()
            outs.kld = kl_divergence(posterior, prior).sum(-1).mean()
            outs.loss = outs.kld  - outs.recon
        else:
            py = self.predict(batch.xc, batch.yc, batch.x, num_samples=num_samples)
            if num_samples is None:
                ll = py.log_prob(batch.y).sum(-1)
            else:
                y = torch.stack([batch.y]*num_samples)
                ll = logmeanexp(py.log_prob(y).sum(-1))
            num_ctx = batch.xc.shape[-2]
            outs.ctx_ll = ll[...,:num_ctx].mean()
            outs.tar_ll = ll[...,num_ctx:].mean()
        return outs

parser = argparse.ArgumentParser()
parser.add_argument('--dim_hid', type=int, default=128)
parser.add_argument('--dim_lat', type=int, default=128)
load = gen_load_func(parser, LANP)

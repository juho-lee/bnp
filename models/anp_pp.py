import argparse
import math

import torch
import torch.nn as nn
from torch.distributions import kl_divergence, Normal
from attrdict import AttrDict

from utils.misc import gen_load_func, logmeanexp

from models.modules import AttEncoder, Encoder, Decoder

class ANPpp(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128,
            dim_lat=128, fixed_var=False):
        super().__init__()
        self.denc = AttEncoder(dim_x=dim_x, dim_y=dim_y, dim_hid=dim_hid)
        self.llenc = AttEncoder(dim_x=dim_x, dim_y=dim_y,
                dim_hid=dim_hid, dim_lat=dim_lat)
        self.glenc = Encoder(dim_x=dim_x, dim_y=dim_y,
                dim_hid=dim_hid, dim_lat=dim_lat)
        self.dec = Decoder(dim_x=dim_x, dim_y=dim_y, dim_enc=dim_hid+2*dim_lat,
                dim_hid=dim_hid, fixed_var=fixed_var)

    def predict(self, xc, yc, xt, pzl=None, pzg=None,  num_samples=None):
        hid = self.denc(xc, yc, xt)
        pzl = self.llenc(xc, yc, xt) if pzl is None else pzl
        pzg = self.glenc(xc, yc) if pzg is None else pzg
        if num_samples is None:
            zl = pzl.rsample()
            zg = pzg.rsample()
        else:
            hid = torch.stack([hid]*num_samples)
            zl = pzl.rsample([num_samples])
            zg = pzg.rsample([num_samples])
            xt = torch.stack([xt]*num_samples)
        zg = torch.stack([zg]*hid.shape[-2], -2)
        return self.dec(torch.cat([hid, zl, zg], -1), xt)

    def forward(self, batch, num_samples=None):
        outs = AttrDict()
        if self.training:
            lprior = self.llenc(batch.xc, batch.yc, batch.x)
            lposterior = self.llenc(batch.x, batch.y, batch.x)
            gprior = self.glenc(batch.xc, batch.yc)
            gposterior = self.glenc(batch.x, batch.y)

            py = self.predict(batch.xc, batch.yc, batch.x,
                    pzl=lposterior, pzg=gposterior)

            outs.recon = py.log_prob(batch.y).sum(-1).mean()
            outs.lkld = kl_divergence(lposterior, lprior).sum(-1).mean()
            outs.gkld = kl_divergence(gposterior, gprior).sum(-1).mean()
            outs.loss = outs.lkld + outs.gkld/batch.x.shape[-2]  - outs.recon
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
parser.add_argument('--fixed_var', '-fv', action='store_true', default=False)
load = gen_load_func(parser, ANPpp)

import argparse
import math

import torch
import torch.nn as nn
from attrdict import AttrDict

from utils.sampling import sample_with_replacement, sample_with_partial_replacement
from utils.misc import gen_load_func, logmeanexp

from models.modules import AttEncoder, Encoder, Decoder

class BANPpp(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128,
            no_bootstrap=False, fixed_var=False):
        super().__init__()
        self.bootstrap = not no_bootstrap
        self.denc = AttEncoder(dim_x=dim_x, dim_y=dim_y, dim_hid=dim_hid)
        self.benc1 = AttEncoder(dim_x=dim_x, dim_y=dim_y, dim_hid=dim_hid)
        self.benc2 = Encoder(dim_x=dim_x, dim_y=dim_y, dim_hid=dim_hid)
        self.dec = Decoder(dim_x=dim_x, dim_y=dim_y, dim_enc=3*dim_hid,
                dim_hid=dim_hid, fixed_var=fixed_var)

    def predict(self, xc, yc, xt, bootstrap=None, num_samples=None):
        bootstrap = self.bootstrap if bootstrap is None else self.bootstrap
        if self.training:
            bxc, byc = sample_with_replacement([xc, yc])
            hid1 = self.denc(bxc, byc, xt)
        else:
            hid1 = self.denc(xc, yc, xt)

        if bootstrap:
            if num_samples is not None:
                hid1 = torch.stack([hid1]*num_samples)
                xt = torch.stack([xt]*num_samples)
            bxc, byc = sample_with_replacement([xc, yc], num_samples=num_samples)
            hid2 = self.benc1(bxc, byc, xt)

            bxc, byc = sample_with_replacement([xc, yc], num_samples=num_samples)
            hid3 = self.benc2(bxc, byc)
        else:
            hid2 = self.benc1(xc, yc)
            hid3 = self.benc2(xc, yc)

        hid3 = torch.stack([hid3]*hid2.shape[-2], -2)
        return self.dec(torch.cat([hid1, hid2, hid3], -1), xt)

    def forward(self, batch, num_samples=None):
        outs = AttrDict()
        if self.training:
            py = self.predict(batch.xc, batch.yc, batch.x)
            outs.ll = py.log_prob(batch.y).sum(-1).mean()
            outs.loss = -outs.ll
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
parser.add_argument('--no_bootstrap', '-nbs', action='store_true', default=False)
parser.add_argument('--fixed_var', '-fv', action='store_true', default=False)
load = gen_load_func(parser, BANPpp)

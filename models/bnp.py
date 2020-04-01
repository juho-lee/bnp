import argparse
import math

import torch
import torch.nn as nn
from attrdict import AttrDict

from utils.misc import gen_load_func, logmeanexp
from utils.sampling import sample_with_replacement, sample_subset

from models.modules import Encoder, Decoder

class BNP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, r_N=1.0):
        super().__init__()
        self.r_N = r_N
        self.denc = Encoder(dim_x=dim_x, dim_y=dim_y, dim_hid=dim_hid)
        self.benc = Encoder(dim_x=dim_x, dim_y=dim_y, dim_hid=dim_hid)
        self.dec = Decoder(dim_x=dim_x, dim_y=dim_y,
                dim_enc=2*dim_hid, dim_hid=dim_hid)

    def predict(self, xc, yc, xt, bootstrap=True, num_samples=None):
        hid1 = self.denc(xc, yc)
        if bootstrap:
            bxc, byc = sample_with_replacement([xc, yc],
                    num_samples=num_samples, r_N=self.r_N)
            hid2 = self.benc(bxc, byc)
            if num_samples is not None:
                hid1 = torch.stack([hid1]*num_samples)
                xt = torch.stack([xt]*num_samples)
        else:
            hid2 = self.benc(xc, yc)
        return self.dec(torch.cat([hid1, hid2], -1), xt)

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

    def validate(self, batch, num_splits=10, num_samples=None):
        log_diffs = []
        for _ in range(num_splits):
            (xcc, ycc), (xct, yct) = sample_subset([batch.xc, batch.yc],
                    r_N=torch.rand(1).item())
            py = self.predict(xcc, ycc, xct, num_samples=num_samples)
            ll = logmeanexp(py.log_prob(torch.stack([yct]*num_samples)).sum(-1))
            py_det = self.predict(xcc, ycc, xct, bootstrap=False)
            ll_det = py_det.log_prob(yct).sum(-1)
            log_diffs.append((ll - ll_det).mean())
        return torch.stack(log_diffs).mean()

parser = argparse.ArgumentParser()
parser.add_argument('--dim_hid', type=int, default=128)
parser.add_argument('--r_N', type=float, default=1.0)
load = gen_load_func(parser, BNP)

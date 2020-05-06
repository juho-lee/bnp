import argparse

import torch
import torch.nn as nn
from torch.distributions import Normal
from attrdict import AttrDict

from models.modules import PoolingEncoder, Decoder

from utils.misc import gen_load_func
from utils.sampling import sample_with_replacement as SWR

class CNP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128):
        super().__init__()
        self.dim_enc = dim_hid
        self.enc = PoolingEncoder(dim_x=dim_x, dim_y=dim_y, dim_hid=dim_hid)
        self.dec = Decoder(dim_x=dim_x, dim_y=dim_y,
                dim_enc=self.dim_enc, dim_hid=dim_hid)

    def encode(self, xc, yc, xt, num_samples=None, mask=None):
        return torch.stack([self.enc(xc, yc, mask=mask)]*xt.shape[-2], -2)

    def predict(self, xc, yc, xt, num_samples=None, mask=None):
        return self.dec(self.encode(xc, yc, xt, mask=mask), xt)

    def forward(self, batch, num_samples=None):
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

parser = argparse.ArgumentParser()
parser.add_argument('--dim_hid', type=int, default=128)
load = gen_load_func(parser, CNP)

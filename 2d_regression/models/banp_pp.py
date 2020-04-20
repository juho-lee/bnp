import argparse
import math

import torch
import torch.nn as nn
from attrdict import AttrDict

from utils.sampling import sample_with_replacement, sample_subset
from utils.misc import gen_load_func, logmeanexp

from models.bnp import BNP
from models.modules import AttEncoder, Encoder, Decoder

class BANPpp(BNP):
    def __init__(self, dim_x=2, dim_y=1, dim_hid=128, r_N=1.0):
        nn.Module.__init__(self)
        self.r_N = r_N
        self.denc = AttEncoder(dim_x=dim_x, dim_y=dim_y,
                dim_hid=dim_hid, self_attn=True)
        self.benc1 = AttEncoder(dim_x=dim_x, dim_y=dim_y,
                dim_hid=dim_hid, self_attn=True)
        self.benc2 = Encoder(dim_x=dim_x, dim_y=dim_y,
                dim_hid=dim_hid, self_attn=True)
        self.dec = Decoder(dim_x=dim_x, dim_y=dim_y,
                dim_enc=3*dim_hid, dim_hid=dim_hid)

    def predict(self, xc, yc, xt, bootstrap=True, num_samples=None):
        if self.training:
            bxc, byc = sample_with_replacement([xc, yc])
            hid1 = self.denc(bxc, byc, xt)
        else:
            hid1 = self.denc(xc, yc, xt)

        if bootstrap:
            if num_samples is not None:
                hid1 = torch.stack([hid1]*num_samples)
                xt = torch.stack([xt]*num_samples)
            bxc, byc = sample_with_replacement([xc, yc],
                    num_samples=num_samples, r_N=self.r_N)
            hid2 = self.benc1(bxc, byc, xt)

            bxc, byc = sample_with_replacement([xc, yc],
                    num_samples=num_samples, r_N=self.r_N)
            hid3 = self.benc2(bxc, byc)
        else:
            hid2 = self.benc1(xc, yc, xt)
            hid3 = self.benc2(xc, yc)

        hid3 = torch.stack([hid3]*hid2.shape[-2], -2)
        return self.dec(torch.cat([hid1, hid2, hid3], -1), xt)

parser = argparse.ArgumentParser()
parser.add_argument('--dim_hid', type=int, default=128)
parser.add_argument('--r_N', type=float, default=1.0)
load = gen_load_func(parser, BANPpp)

import argparse

import torch
import torch.nn as nn
from torch.distributions import kl_divergence, Normal
from attrdict import AttrDict

from utils.misc import gen_load_func, logmeanexp, stack

from models.gp.np import NP
from models.modules import *

from utils.sampling import sample_with_replacement as SWR

class ANP(NP):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128):
        nn.Module.__init__(self)
        self.dim_enc = dim_hid + dim_lat
        self.denc = CrossAttnEncoder(dim_x=dim_x, dim_y=dim_y, dim_hid=dim_hid)
        self.lenc = PoolingEncoder(dim_x=dim_x, dim_y=dim_y,
                dim_hid=dim_hid, dim_lat=dim_lat, self_attn=True)
        self.dec = Decoder(dim_x=dim_x, dim_y=dim_y,
                dim_enc=self.dim_enc, dim_hid=dim_hid)

    def encode(self, xc, yc, xt, z=None, num_samples=None, mask=None):
        if z is None:
            pz = self.lenc(xc, yc, mask=mask)
            z = pz.rsample() if num_samples is None \
                    else pz.rsample([num_samples])
        hid = stack(self.denc(xc, yc, xt, mask=mask), num_samples)
        z = stack(z, hid.shape[-2], -2)
        return torch.cat([hid, z], -1)

    def predict(self, xc, yc, xt, z=None, num_samples=None, mask=None):
        encoded = self.encode(xc, yc, xt, z=z, num_samples=num_samples, mask=mask)
        return self.dec(encoded, stack(xt, num_samples))

parser = argparse.ArgumentParser()
parser.add_argument('--dim_hid', type=int, default=128)
parser.add_argument('--dim_lat', type=int, default=128)
load = gen_load_func(parser, ANP)

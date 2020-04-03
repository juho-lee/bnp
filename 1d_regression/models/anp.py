import argparse
import math

import torch
import torch.nn as nn
from torch.distributions import kl_divergence, Normal
from attrdict import AttrDict

from utils.misc import gen_load_func, logmeanexp

from models.np import NP
from models.modules import AttEncoder, Encoder, Decoder

class ANP(NP):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128):
        nn.Module.__init__(self)
        self.denc = AttEncoder(dim_x=dim_x, dim_y=dim_y, dim_hid=dim_hid)
        self.lenc = Encoder(dim_x=dim_x, dim_y=dim_y,
                dim_hid=dim_hid, dim_lat=dim_lat)
        self.dec = Decoder(dim_x=dim_x, dim_y=dim_y,
                dim_enc=dim_hid+dim_lat, dim_hid=dim_hid)

    def predict(self, xc, yc, xt, pz=None, num_samples=None):
        hid = self.denc(xc, yc, xt)
        pz = self.lenc(xc, yc) if pz is None else pz
        if num_samples is None:
            z = pz.rsample()
        else:
            hid = torch.stack([hid]*num_samples)
            xt = torch.stack([xt]*num_samples)
            z = pz.rsample([num_samples])
        z = torch.stack([z]*hid.shape[-2], -2)
        return self.dec(torch.cat([hid, z], -1), xt)

parser = argparse.ArgumentParser()
parser.add_argument('--dim_hid', type=int, default=128)
parser.add_argument('--dim_lat', type=int, default=128)
load = gen_load_func(parser, ANP)

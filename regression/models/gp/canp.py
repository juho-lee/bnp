import argparse

import torch
import torch.nn as nn
from torch.distributions import Normal
from attrdict import AttrDict

from utils.misc import gen_load_func, stack
from utils.sampling import sample_with_replacement as SWR

from models.gp.cnp import CNP
from models.modules import CrossAttnEncoder, Decoder, PoolingEncoder

class CANP(CNP):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128):
        nn.Module.__init__(self)
        self.dim_enc = dim_hid
        self.enc = CrossAttnEncoder(dim_x=dim_x, dim_y=dim_y,
                dim_hid=dim_hid, self_attn=True)
        self.dec = Decoder(dim_x=dim_x, dim_y=dim_y,
                dim_enc=self.dim_enc, dim_hid=dim_hid)

    def encode(self, xc, yc, xt, num_samples=None, mask=None):
        return self.enc(xc, yc, xt, mask=mask)

    def predict(self, xc, yc, xt, num_samples=None, mask=None):
        return self.dec(self.encode(xc, yc, xt, mask=mask), xt)

parser = argparse.ArgumentParser()
parser.add_argument('--dim_hid', type=int, default=128)
load = gen_load_func(parser, CANP)

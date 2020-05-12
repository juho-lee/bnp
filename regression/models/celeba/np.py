import argparse

import torch
import torch.nn as nn
from torch.distributions import kl_divergence, Normal
from attrdict import AttrDict

from utils.misc import gen_load_func, logmeanexp, stack

from models.gp.np import NP as Base
from models.modules import PoolingEncoder, Decoder

class NP(Base):
    def __init__(self, dim_x=2, dim_y=3, dim_hid=128, dim_lat=128):
        nn.Module.__init__(self)
        self.dim_enc = dim_hid+dim_lat
        self.denc = PoolingEncoder(dim_x=dim_x, dim_y=dim_y, dim_hid=dim_hid,
                pre_depth=6, post_depth=3)
        self.lenc = PoolingEncoder(dim_x=dim_x, dim_y=dim_y,
                dim_hid=dim_hid, dim_lat=dim_lat,
                pre_depth=6, post_depth=3)
        self.dec = Decoder(dim_x=dim_x, dim_y=dim_y,
                dim_enc=self.dim_enc, dim_hid=dim_hid,
                depth=5)

parser = argparse.ArgumentParser()
parser.add_argument('--dim_hid', type=int, default=128)
parser.add_argument('--dim_lat', type=int, default=128)
load = gen_load_func(parser, NP)

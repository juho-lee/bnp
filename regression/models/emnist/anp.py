import argparse

import torch
import torch.nn as nn
from torch.distributions import kl_divergence, Normal
from attrdict import AttrDict

from utils.misc import gen_load_func, logmeanexp, stack

from models.gp.anp import ANP as Base
from models.modules import *

class ANP(Base):
    def __init__(self, dim_x=2, dim_y=1, dim_hid=128, dim_lat=128):
        nn.Module.__init__(self)
        self.dim_enc = dim_hid+dim_lat
        self.denc = CrossAttnEncoder(dim_x=dim_x, dim_y=dim_y, dim_hid=dim_hid,
                v_depth=5, qk_depth=3)
        self.lenc = PoolingEncoder(dim_x=dim_x, dim_y=dim_y,
                dim_hid=dim_hid, dim_lat=dim_lat, self_attn=True,
                pre_depth=5, post_depth=3)
        self.dec = Decoder(dim_x=dim_x, dim_y=dim_y,
                dim_enc=self.dim_enc, dim_hid=dim_hid,
                depth=4)

parser = argparse.ArgumentParser()
parser.add_argument('--dim_hid', type=int, default=128)
parser.add_argument('--dim_lat', type=int, default=128)
load = gen_load_func(parser, ANP)

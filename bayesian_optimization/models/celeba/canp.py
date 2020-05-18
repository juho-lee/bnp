import argparse

import torch
import torch.nn as nn
from torch.distributions import kl_divergence, Normal
from attrdict import AttrDict

from utils.misc import gen_load_func, logmeanexp, stack

from models.gp.canp import CANP as Base
from models.modules import *

class CANP(Base):
    def __init__(self, dim_x=2, dim_y=3, dim_hid=128, dim_lat=128):
        nn.Module.__init__(self)
        self.dim_enc = dim_hid
        self.enc = CrossAttnEncoder(dim_x=dim_x, dim_y=dim_y, dim_hid=dim_hid,
                v_depth=6, qk_depth=3)
        self.dec = Decoder(dim_x=dim_x, dim_y=dim_y,
                dim_enc=dim_hid, dim_hid=dim_hid, depth=5)

parser = argparse.ArgumentParser()
parser.add_argument('--dim_hid', type=int, default=128)
load = gen_load_func(parser, CANP)

import argparse

import torch
import torch.nn as nn
from torch.distributions import Normal
from attrdict import AttrDict

from utils.misc import gen_load_func, stack
from utils.sampling import sample_with_replacement as SWR

from models.gp.cnp import CNP as Base
from models.modules import PoolingEncoder, Decoder

class CNP(Base):
    def __init__(self, dim_x=2, dim_y=3, dim_hid=128):
        super().__init__()
        self.dim_enc = dim_hid
        self.enc = PoolingEncoder(dim_x=dim_x, dim_y=dim_y, dim_hid=dim_hid,
                pre_depth=6, post_depth=3)
        self.dec = Decoder(dim_x=dim_x, dim_y=dim_y,
                dim_enc=dim_hid, dim_hid=dim_hid, depth=5)

parser = argparse.ArgumentParser()
parser.add_argument('--dim_hid', type=int, default=128)
load = gen_load_func(parser, CNP)

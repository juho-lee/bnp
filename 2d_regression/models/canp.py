import argparse

import torch
import torch.nn as nn
from attrdict import AttrDict

from utils.misc import gen_load_func

from models.cnp import CNP
from models.modules import AttEncoder, Decoder

class CANP(CNP):
    def __init__(self, dim_x=2, dim_y=1, dim_hid=128):
        nn.Module.__init__(self)
        self.enc = AttEncoder(dim_x=dim_x, dim_y=dim_y,
                dim_hid=dim_hid, self_attn=True)
        self.dec = Decoder(dim_x=dim_x, dim_y=dim_y,
                dim_enc=dim_hid, dim_hid=dim_hid)

    def predict(self, xc, yc, xt, num_samples=None):
        return self.dec(self.enc(xc, yc, xt), xt)

parser = argparse.ArgumentParser()
parser.add_argument('--dim_hid', type=int, default=128)
load = gen_load_func(parser, CANP)

import torch
import torch.nn as nn
from attrdict import AttrDict

from models.modules import DetEncoder, Decoder

class CNP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, fixed_var=False):
        super().__init__()
        self.enc = DetEncoder(dim_x, dim_y, dim_hid)
        self.dec = Decoder(dim_x, dim_y, dim_hid, dim_hid, fixed_var)

    def forward(self, batch, num_samples=None):
        outs = AttrDict()
        hid = self.enc(batch.xc, batch.yc)
        if self.training:
            py = self.dec(hid, batch.x)
            outs.ll = py.log_prob(batch.y).sum(-1).mean()
            outs.loss = -outs.ll
        else:
            py = self.dec(hid, batch.xt)
            outs.pred_ll = py.log_prob(batch.yt).sum(-1).mean()
        return outs

    def predict(self, xc, yc, xt, num_samples=None):
        hid = self.enc(xc, yc)
        py = self.dec(hid, xt)
        return py.mean, py.scale

def load(args):
    return CNP(fixed_var=args.fixed_var)

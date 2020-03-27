import torch
import torch.nn as nn
from attrdict import AttrDict
import math

from models.modules import DetEncoder, Decoder
from models.bootstrap import sample_bootstrap

class CNP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, fixed_var=False):
        super().__init__()
        self.enc = DetEncoder(dim_x, dim_y, dim_hid)
        self.dec = Decoder(dim_x, dim_y, dim_hid, dim_hid, fixed_var)

    def predict(self, xc, yc, xt, num_samples=None, r_bs=0.0):
        K = num_samples or 1
        if r_bs > 0:
            bxc, byc = sample_bootstrap(xc, yc, num_samples=K, r_bs=r_bs)
            xt = torch.stack([xt]*K)
            hid = self.enc(bxc, byc)
        else:
            hid = self.enc(xc, yc)
        return self.dec(hid, xt)

    def forward(self, batch, num_samples=None, r_bs=0.0):
        outs = AttrDict()
        if self.training:
            hid = self.enc(batch.xc, batch.yc)
            py = self.dec(hid, batch.x)
            outs.ll = py.log_prob(batch.y).sum(-1).mean()
            outs.loss = -outs.ll
        else:
            K = num_samples or 1
            py = self.predict(batch.xc, batch.yc, batch.xc,
                    num_samples=K, r_bs=r_bs)
            outs.ctx_ll = py.log_prob(batch.yc).sum(-1).mean()

            py = self.predict(batch.xc, batch.yc, batch.xt,
                    num_samples=K, r_bs=r_bs)
            outs.pred_ll = py.log_prob(batch.yt).sum(-1).mean()
        return outs

def load(args):
    return CNP(fixed_var=args.fixed_var)

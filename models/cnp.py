import torch
import torch.nn as nn
from attrdict import AttrDict
import math

from models.modules import DetEncoder, Decoder
from models.bootstrap import sample_bootstrap

class CNP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128,
            fixed_var=False, r_bs=0.0):
        super().__init__()
        self.r_bs = r_bs
        self.enc = DetEncoder(dim_x, dim_y, dim_hid)
        self.dec = Decoder(dim_x, dim_y, dim_hid, dim_hid, fixed_var)

    def predict(self, xc, yc, xt, num_samples=None):
        K = num_samples or 1
        if self.r_bs > 0.0:
            bxc, byc = sample_bootstrap(xc, yc, r_bs=self.r_bs, num_samples=K)
            return self.dec(self.enc(bxc, byc), torch.stack([xt]*K))
        else:
            return self.dec(self.enc(xc, yc), xt)

    def forward(self, batch, num_samples=None):
        outs = AttrDict()
        if self.training:
            if self.r_bs > 0:
                bxc, byc = sample_bootstrap(batch.xc, batch.yc, r_bs=self.r_bs)
                hid = self.enc(bxc, byc)
            else:
                hid = self.enc(batch.xc, batch.yc)
            py = self.dec(hid, batch.x)
            outs.ll = py.log_prob(batch.y).sum(-1).mean()
            outs.loss = -outs.ll
        else:
            K = num_samples or 1
            py = self.predict(batch.xc, batch.yc, batch.xt, num_samples=K)
            if self.r_bs > 0:
                yt = torch.stack([batch.yt]*K)
                pred_ll = py.log_prob(yt).sum(-1).logsumexp(0) - math.log(K)
                outs.pred_ll = pred_ll.mean()
            else:
                outs.pred_ll = py.log_prob(batch.yt).sum(-1).mean()
        return outs

def load(args):
    return CNP(fixed_var=args.fixed_var, r_bs=args.r_bs)

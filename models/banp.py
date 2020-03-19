import torch
import torch.nn as nn
from attrdict import AttrDict
import math
import numpy as np

from models.modules import AttDetEncoder, LatEncoder, Decoder
from models.bootstrap import sample_bootstrap

class BANP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128,
            fixed_var=False, r_bs=0.2):
        super().__init__()
        self.denc = AttDetEncoder(dim_x, dim_y, dim_hid)
        self.benc = LatEncoder(dim_x, dim_y, dim_hid, dim_lat, rand=False)
        self.dec = Decoder(dim_x, dim_y, dim_hid+dim_lat, dim_hid, fixed_var)
        self.r_bs = r_bs

    def predict(self, xc, yc, xt, num_samples=None):
        K = num_samples or 1
        if self.r_bs > 0.0:
            xt = torch.stack([xt]*K)
            bxc, byc = sample_bootstrap(xc, yc, r_bs=self.r_bs, num_samples=K)
            hid = self.denc(bxc, byc, xt)
        else:
            hid = self.denc(xc, yc, xt)
            hid = torch.stack([hid]*K)
            xt = torch.stack([xt]*K)

        bxc, byc = sample_bootstrap(xc, yc, r_bs=1.0, num_samples=K)
        z = self.benc(bxc, byc)

        encoded = torch.cat([hid, torch.stack([z]*hid.shape[-2], -2)], -1)
        return self.dec(encoded, xt)

    def forward(self, batch, num_samples=None):
        outs = AttrDict()
        if self.training:
            if self.r_bs > 0.0:
                bxc, byc = sample_bootstrap(batch.xc, batch.yc, r_bs=self.r_bs)
                hid = self.denc(bxc, byc, batch.x)
            else:
                hid = self.denc(batch.xc, batch.yc, batch.x)

            bxc, byc = sample_bootstrap(batch.xc, batch.yc, r_bs=1.0)
            z = self.benc(bxc, byc)

            encoded = torch.cat([hid, torch.stack([z]*hid.shape[-2], -2)], -1)
            py = self.dec(encoded, batch.x)
            outs.ll = py.log_prob(batch.y).sum(-1).mean()
            outs.loss = -outs.ll
        else:
            K = num_samples or 1
            py = self.predict(batch.xc, batch.yc, batch.xt, num_samples=K)
            yt = torch.stack([batch.yt]*K)
            pred_ll = py.log_prob(yt).sum(-1).logsumexp(0) - math.log(K)
            outs.pred_ll = pred_ll.mean()
        return outs

def load(args):
    return BANP(fixed_var=args.fixed_var, r_bs=args.r_bs)

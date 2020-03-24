import torch
import torch.nn as nn
import torch.nn.functional as F
from attrdict import AttrDict
import math
import numpy as np

from models.modules import AttDetEncoder, LatEncoder, Decoder, \
        MultiplicativeInteraction
from models.bootstrap import sample_bootstrap, random_split

class BANP2(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128,
            dim_enc=32, fixed_var=False):
        super().__init__()
        self.denc = AttDetEncoder(dim_x, dim_y, dim_hid)
        self.benc = LatEncoder(dim_x, dim_y, dim_hid, dim_lat, rand=False)
        self.mi = MultiplicativeInteraction(dim_hid, dim_lat, dim_enc)
        self.dec = Decoder(dim_x, dim_y, dim_enc, dim_hid, fixed_var)

    def predict(self, xc, yc, xt, num_samples=None, r_bs=0.0):
        K = num_samples or 1
        if r_bs == 0:
            hid = self.denc(xc, yc, xt)
            hid = torch.stack([hid]*K)
            xt = torch.stack([xt]*K)
        else:
            bxc, byc = sample_bootstrap(xc, yc, r_bs=r_bs, num_samples=K)
            xt = torch.stack([xt]*K)
            hid = self.denc(bxc, byc, xt)
        bxc, byc = sample_bootstrap(xc, yc, r_bs=1.0, num_samples=K)
        z = self.benc(bxc, byc)
        z = torch.stack([z]*hid.shape[-2], -2)
        encoded = self.mi(hid, z)
        return self.dec(encoded, xt)

    def forward(self, batch, num_samples=None, r_bs=0.0):
        outs = AttrDict()
        if self.training:
            bxc, byc = sample_bootstrap(batch.xc, batch.yc)
            hid = self.denc(bxc, byc, batch.x)
            bxc, byc = sample_bootstrap(batch.xc, batch.yc, r_bs=1.0)
            z = self.benc(bxc, byc)
            z = torch.stack([z]*hid.shape[-2], -2)
            encoded = self.mi(hid, z)
            py = self.dec(encoded, batch.x)

            outs.ll = py.log_prob(batch.y).sum(-1).mean()
            outs.loss = -outs.ll
        else:
            K = num_samples or 1
            py = self.predict(batch.xc, batch.yc, batch.xt,
                    num_samples=K, r_bs=r_bs)
            yt = torch.stack([batch.yt]*K)
            pred_ll = py.log_prob(yt).sum(-1).logsumexp(0) - math.log(K)
            outs.pred_ll = pred_ll.mean()
        return outs

def load(args):
    return BANP2(fixed_var=args.fixed_var)

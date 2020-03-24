import torch
import torch.nn as nn
from attrdict import AttrDict
import math

from models.modules import DetEncoder, LatEncoder, Decoder
from models.bootstrap import sample_bootstrap

class BNP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128, fixed_var=False):
        super().__init__()
        self.denc = DetEncoder(dim_x, dim_y, dim_hid)
        self.benc = LatEncoder(dim_x, dim_y, dim_hid, dim_lat, rand=False)
        self.dec = Decoder(dim_x, dim_y, dim_hid+dim_lat, dim_hid, fixed_var)

    def predict(self, xc, yc, xt, num_samples=None, r_bs=0.0):
        K = num_samples or 1
        if r_bs > 0.0:
            bxc, byc = sample_bootstrap(xc, yc, r_bs=r_bs, num_samples=K)
            hid = self.denc(bxc, byc)
        else:
            hid = self.denc(xc, yc)
            hid = torch.stack([hid]*K)
        bxc, byc = sample_bootstrap(xc, yc, num_samples=K)
        z = self.benc(bxc, byc)

        encoded = torch.cat([hid, z], -1)
        return self.dec(encoded, torch.stack([xt]*K))

        encoded = torch.cat([hid, torch.stack([z]*hid.shape[-2], -2)], -1)
        return self.dec(encoded, xt)

    def forward(self, batch, num_samples=None, r_bs=0.0):
        outs = AttrDict()
        if self.training:
            bxc, byc = sample_bootstrap(batch.xc, batch.yc)
            hid = self.denc(bxc, byc)
            bxc, byc = sample_bootstrap(batch.xc, batch.yc)
            z = self.benc(bxc, byc)
            encoded = torch.cat([hid, z], -1)
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
    return BNP(fixed_var=args.fixed_var)

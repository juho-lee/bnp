import torch
import torch.nn as nn
from attrdict import AttrDict
import math

from models.modules import AttDetEncoder, LatEncoder, Decoder
from models.bootstrap import sample_bootstrap

class BANP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128,
            fixed_var=False, r_fixed=0.8):
        super().__init__()
        self.denc = AttDetEncoder(dim_x, dim_y, dim_hid)
        self.benc = LatEncoder(dim_x, dim_y, dim_hid, dim_lat, rand=False)
        self.dec = Decoder(dim_x, dim_y, dim_hid+dim_lat, dim_hid, fixed_var)
        self.r_fixed = r_fixed

    def det_path(self, xc, yc, xt, num_samples=None):
        if self.training:
            bs_x, bs_y = sample_bootstrap(xc, yc)
            return self.denc(bs_x, bs_y, xt)
        else:
            K = num_samples or 1
            K_fixed = int(K * self.r_fixed)
            K_rand = K - K_fixed
            if K_rand == 0:
                hid = self.denc(xc, yc, xt)
                return torch.stack([hid]*K)
            elif K_fixed == 0:
                bs_x, bs_y = sample_bootstrap(xc, yc, num_samples=K)
                return self.denc(bs_x, bs_y, torch.stack([xt]*K))
            else:
                hid_fixed = torch.stack([self.denc(xc, yc, xt)]*K_fixed)
                bs_x, bs_y = sample_bootstrap(xc, yc, num_samples=K_rand)
                hid_rand = self.denc(bs_x, bs_y, torch.stack([xt]*K_rand))
                return torch.cat([hid_fixed, hid_rand])

    def forward(self, batch, num_samples=None):
        outs = AttrDict()
        if self.training:
            # det path
            hid = self.det_path(batch.xc, batch.yc, batch.x)

            # lat path
            bs_x, bs_y = sample_bootstrap(batch.xc, batch.yc)
            z = self.benc(bs_x, bs_y)
            z = torch.stack([z]*hid.shape[-2], -2)

            py = self.dec(torch.cat([hid, z], -1), batch.x)
            outs.ll = py.log_prob(batch.y).sum(-1).mean()
            outs.loss = -outs.ll
        else:
            K = num_samples or 1

            # det path
            hid = self.det_path(batch.xc, batch.yc, batch.xt, num_samples=K)

            # lat path
            bs_x, bs_y = sample_bootstrap(batch.xc, batch.yc, num_samples=K)
            z = self.benc(bs_x, bs_y)
            z = torch.stack([z]*hid.shape[-2], -2)

            xt = torch.stack([batch.xt]*K)
            yt = torch.stack([batch.yt]*K)
            py = self.dec(torch.cat([hid, z], -1), xt)
            pred_ll = py.log_prob(yt).sum(-1).logsumexp(0) - math.log(K)
            outs.pred_ll = pred_ll.mean()
        return outs

    def predict(self, xc, yc, xt, num_samples=None):
        K = num_samples or 1
        # det path
        hid = self.det_path(xc, yc, xt, num_samples=K)

        # lat path
        bs_x, bs_y = sample_bootstrap(xc, yc, num_samples=K)
        z = self.benc(bs_x, bs_y)
        z = torch.stack([z]*hid.shape[-2], -2)

        xt = torch.stack([xt]*K)
        py = self.dec(torch.cat([hid, z], -1), xt)
        return py.mean.squeeze(0), py.scale.squeeze(0)

def load(args):
    return BANP(fixed_var=args.fixed_var)

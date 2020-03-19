import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from attrdict import AttrDict
import math

from models.modules import DetEncoder, LatEncoder, Decoder
from models.bootstrap import sample_bootstrap

class NP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128,
            fixed_var=False, r_bs=0.0):
        super().__init__()
        self.r_bs = r_bs
        self.denc = DetEncoder(dim_x, dim_y, dim_hid)
        self.lenc = LatEncoder(dim_x, dim_y, dim_hid, dim_lat)
        self.dec = Decoder(dim_x, dim_y, dim_hid+dim_lat, dim_hid, fixed_var)

    def predict(self, xc, yc, xt, num_samples=None):
        K = num_samples or 1
        if self.r_bs > 0.0:
            bxc, byc = sample_bootstrap(xc, yc, r_bs=self.r_bs, num_samples=K)
            hid = self.denc(bxc, byc)
            prior = self.lenc(bxc, byc)
            z = prior.rsample()
        else:
            hid = self.denc(xc, yc)
            hid = torch.stack([hid]*K)
            prior = self.lenc(xc, yc)
            z = prior.rsample([K])

        encoded = torch.cat([hid, z], -1)
        return self.dec(encoded, torch.stack([xt]*K))

    def forward(self, batch, num_samples=None):
        outs = AttrDict()
        if self.training:
            if self.r_bs > 0:
                bxc, byc = sample_bootstrap(batch.xc, batch.yc, r_bs=self.r_bs)
                prior = self.lenc(bxc, byc)
                hid = self.denc(bxc, byc)
                bx, by = sample_bootstrap(batch.x, batch.y, r_bs=self.r_bs)
                posterior = self.lenc(bx, by)
            else:
                prior = self.lenc(batch.xc, batch.yc)
                hid = self.denc(batch.xc, batch.yc)
                posterior = self.lenc(batch.x, batch.y)

            z = posterior.rsample()
            encoded = torch.cat([hid, z], -1)
            py = self.dec(encoded, batch.x)
            outs.recon = py.log_prob(batch.y).sum(-1).mean()
            outs.kld = kl_divergence(posterior, prior).sum(-1).mean()
            outs.loss = outs.kld / batch.x.shape[-2] - outs.recon
        else:
            K = num_samples or 1
            py = self.predict(batch.xc, batch.yc, batch.xt, num_samples=K)
            yt = torch.stack([batch.yt]*K)
            pred_ll = py.log_prob(yt).sum(-1).logsumexp(0) - math.log(K)
            outs.pred_ll = pred_ll.mean()
        return outs

def load(args):
    return NP(fixed_var=args.fixed_var, r_bs=args.r_bs)

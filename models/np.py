import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from attrdict import AttrDict
import math

from models.modules import DetEncoder, LatEncoder, Decoder
from models.bootstrap import sample_bootstrap

class NP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128,
            fixed_var=False):
        super().__init__()
        self.denc = DetEncoder(dim_x, dim_y, dim_hid)
        self.lenc = LatEncoder(dim_x, dim_y, dim_hid, dim_lat)
        self.dec = Decoder(dim_x, dim_y, dim_hid+dim_lat, dim_hid, fixed_var)

    def predict(self, xc, yc, xt, num_samples=None, r_bs=0.0):
        K = num_samples or 1
        if r_bs > 0:
            bxc, byc = sample_bootstrap(xc, yc, r_bs=r_bs, num_samples=K)
            hid = self.denc(bxc, byc)
        else:
            hid = self.denc(xc, yc)
        hid = torch.stack([hid]*K)
        prior = self.lenc(xc, yc)
        z = prior.rsample([K])
        encoded = torch.cat([hid, z], -1)
        return self.dec(encoded, torch.stack([xt]*K))

    def forward(self, batch, num_samples=None, r_bs=0.0):
        outs = AttrDict()
        if self.training:
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
            py = self.predict(batch.xc, batch.yc, batch.xc, num_samples=K, r_bs=r_bs)
            yc = torch.stack([batch.yc]*K)
            ctx_ll = py.log_prob(yc).sum(-1).logsumexp(0) - math.log(K)
            outs.ctx_ll = ctx_ll.mean()

            py = self.predict(batch.xc, batch.yc, batch.xt, num_samples=K, r_bs=r_bs)
            yt = torch.stack([batch.yt]*K)
            pred_ll = py.log_prob(yt).sum(-1).logsumexp(0) - math.log(K)
            outs.pred_ll = pred_ll.mean()
        return outs

def load(args):
    return NP(fixed_var=args.fixed_var)

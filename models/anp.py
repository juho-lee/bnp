import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from attrdict import AttrDict
import math

from models.modules import AttDetEncoder, LatEncoder, Decoder, LatEncoder, \
        MultiplicativeInteraction
from models.bootstrap import sample_bootstrap

class ANP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128,
            dim_enc=32, fixed_var=False):
        super().__init__()
        self.denc = AttDetEncoder(dim_x, dim_y, dim_hid)
        self.lenc = LatEncoder(dim_x, dim_y, dim_hid, dim_lat)
        self.mi = MultiplicativeInteraction(dim_hid, dim_lat, dim_enc)
        self.dec = Decoder(dim_x, dim_y, dim_enc, dim_hid, fixed_var)

    def predict(self, xc, yc, xt, num_samples=None, r_bs=0.0):
        K = num_samples or 1
        if r_bs > 0:
            bxc, byc = sample_bootstrap(xc, yc, num_samples=K, r_bs=r_bs)
            xt = torch.stack([xt]*K)
            hid = self.denc(bxc, byc, xt)
        else:
            hid = self.denc(xc, yc, xt)
            hid = torch.stack([hid]*K)
            xt = torch.stack([xt]*K)
        prior = self.lenc(xc, yc)
        z = prior.rsample([K])
        z = torch.stack([z]*hid.shape[-2], -2)
        encoded = self.mi(hid, z)
        return self.dec(encoded, xt)

    def forward(self, batch, num_samples=None, r_bs=0.0):
        outs = AttrDict()
        if self.training:
            prior = self.lenc(batch.xc, batch.yc)
            hid = self.denc(batch.xc, batch.yc, batch.x)
            posterior = self.lenc(batch.x, batch.y)
            z = posterior.rsample()
            z = torch.stack([z]*hid.shape[-2], -2)
            encoded = self.mi(hid, z)
            py = self.dec(encoded, batch.x)
            outs.recon = py.log_prob(batch.y).sum(-1).mean()
            outs.kld = kl_divergence(posterior, prior).sum(-1).mean()
            outs.loss = outs.kld / batch.x.shape[-2] - outs.recon
        else:
            K = num_samples or 1
            py = self.predict(batch.xc, batch.yc, batch.xt,
                    num_samples=K, r_bs=r_bs)
            yt = torch.stack([batch.yt]*K)
            pred_ll = py.log_prob(yt).sum(-1).logsumexp(0) - math.log(K)
            outs.pred_ll = pred_ll.mean()
        return outs

def load(args):
    return ANP(fixed_var=args.fixed_var)

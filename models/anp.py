import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from attrdict import AttrDict
import math

from models.cnp import Decoder
from models.np import LatentEncoder
from models.canp import DeterministicEncoder

class ANP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128, fixed_var=False):
        super().__init__()
        self.denc = DeterministicEncoder(dim_x, dim_y, dim_hid)
        self.lenc = LatentEncoder(dim_x, dim_y, dim_hid, dim_lat)
        self.dec = Decoder(dim_x, dim_y, dim_hid+dim_lat, dim_hid, fixed_var)

    def forward(self, batch, num_samples=None):
        outs = AttrDict()
        prior = self.lenc(batch.xc, batch.yc)
        if self.training:
            hid = self.denc(batch.xc, batch.yc, batch.x)
            posterior = self.lenc(batch.x, batch.y)
            z = posterior.rsample()
            z = torch.stack([z]*hid.shape[-2], -2)
            py = self.dec(torch.cat([hid, z], -1), batch.x)
            outs.recon = py.log_prob(batch.y).sum(-1).mean()
            outs.kld = kl_divergence(posterior, prior).sum(-1).mean()
            outs.loss = outs.kld / batch.x.shape[-2] - outs.recon
        else:
            K = num_samples or 1
            hid = self.denc(batch.xc, batch.yc, batch.xt)
            hid = torch.stack([hid]*K)
            z = prior.rsample([K])
            z = torch.stack([z]*hid.shape[-2], -2)
            xt = torch.stack([batch.xt]*K)
            yt = torch.stack([batch.yt]*K)
            py = self.dec(torch.cat([hid, z], -1), xt)
            pred_ll = py.log_prob(yt).sum(-1).logsumexp(0) - math.log(K)
            outs.pred_ll = pred_ll.mean()
        return outs

    def predict(self, xc, yc, xt, num_samples=None):
        prior = self.lenc(xc, yc)
        hid = self.denc(xc, yc, xt)
        if num_samples is not None and num_samples > 1:
            hid = torch.stack([hid]*num_samples)
            z = prior.rsample([num_samples])
            xt = torch.stack([xt]*num_samples)
        else:
            z = prior.rsample()
        z = torch.stack([z]*hid.shape[-2], -2)
        py = self.dec(torch.cat([hid, z], -1), xt)
        return py.mean, py.scale

def load(args):
    return ANP(fixed_var=args.fixed_var)

#class BANP(nn.Module):
#    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=125, fixed_var=False):
#        super().__init__()
#        self.denc = DeterministicEncoder(dim_x, dim_y, dim_hid)
#        self.benc = BootstrapEncoder(dim_x, dim_y, dim_hid, dim_lat)
#        self.dec = Decoder(dim_x, dim_y, dim_hid+dim_lat, dim_hid, fixed_var)
#
#    def forward(self, batch, num_samples=None):
#        outs = AttrDict()
#        if self.training:
#            hid = self.denc(batch.xc, batch.yc, batch.x)
#            #r = 0.5 + 0.7 * torch.rand([1]).item()
#            z = self.benc(batch.x, batch.y)
#            z = torch.stack([z]*hid.shape[-2], -2)
#            py = self.dec(torch.cat([hid, z], -1), batch.x)
#            outs.ll = py.log_prob(batch.y).sum(-1).mean()
#            outs.loss = -outs.ll
#        else:
#            K = num_samples or 1
#            hid = self.denc(batch.xc, batch.yc, batch.xt)
#            hid = torch.stack([hid]*K)
#            #r = 0.5 + 0.7 * torch.rand([1]).item()
#            z = self.benc(batch.xc, batch.yc, num_samples=K)
#            z = torch.stack([z]*hid.shape[-2], -2)
#            xt = torch.stack([batch.xt]*K)
#            yt = torch.stack([batch.yt]*K)
#            py = self.dec(torch.cat([hid, z], -1), xt)
#            pred_ll = py.log_prob(yt).sum(-1).logsumexp(0) - math.log(K)
#            outs.pred_ll = pred_ll.mean()
#        return outs
#
#    def predict(self, xc, yc, xt, num_samples=None):
#        hid = self.denc(xc, yc, xt)
#        z = self.benc(xc, yc, num_samples=num_samples)
#        if num_samples is not None and num_samples > 1:
#            hid = torch.stack([hid]*num_samples)
#            xt = torch.stack([xt]*num_samples)
#        z = torch.stack([z]*hid.shape[-2], -2)
#        py = self.dec(torch.cat([hid, z], -1), xt)
#        return py.mean, py.scale

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from attrdict import AttrDict
import math

class DeterministicEncoder(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128):
        super().__init__()
        self.mlp = nn.Sequential(
                nn.Linear(dim_x+dim_y, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid))

    def forward(self, x, y):
        return self.mlp(torch.cat([x, y], -1)).mean(-2)

class Decoder(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_enc=128, dim_hid=128):
        super().__init__()
        self.mlp = nn.Sequential(
                nn.Linear(dim_x+dim_enc, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, 2*dim_y))

    def forward(self, encoded, x):
        if encoded.dim() < x.dim():
            encoded = torch.stack([encoded]*x.shape[-2], -2)
        mu, sigma = self.mlp(torch.cat([encoded, x], -1)).chunk(2, -1)
        sigma = 0.1 + 0.9 * F.softplus(sigma)
        return Normal(mu, sigma)

class CNP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128):
        super().__init__()
        self.enc = DeterministicEncoder(dim_x, dim_y, dim_hid)
        self.dec = Decoder(dim_x, dim_y, dim_hid, dim_hid)

    def forward(self, batch, num_samples=None):
        outs = AttrDict()
        hid = self.enc(batch.xc, batch.yc)
        if self.training:
            py = self.dec(hid, batch.x)
            outs.ll = py.log_prob(batch.y).sum(-1).mean()
            outs.loss = -outs.ll
        else:
            py = self.dec(hid, batch.xt)
            outs.pred_ll = py.log_prob(batch.yt).sum(-1).mean()
        return outs

    def predict(self, xc, yc, xt, num_samples=None):
        hid = self.enc(xc, yc)
        py = self.dec(hid, xt)
        return py.mean, py.scale

class LatentEncoder(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128):
        super().__init__()
        self.pre_mlp = nn.Sequential(
                nn.Linear(dim_x+dim_y, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid))
        self.post_mlp = nn.Sequential(
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, 2*dim_lat))

    def forward(self, x, y):
        hid = self.pre_mlp(torch.cat([x, y], -1)).mean(-2)
        mu, sigma = self.post_mlp(hid).chunk(2, -1)
        sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
        return Normal(mu, sigma)

class NP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128):
        super().__init__()
        self.denc = DeterministicEncoder(dim_x, dim_y, dim_hid)
        self.lenc = LatentEncoder(dim_x, dim_y, dim_hid, dim_lat)
        self.dec = Decoder(dim_x, dim_y, dim_hid+dim_lat, dim_hid)

    def forward(self, batch, num_samples=None):
        outs = AttrDict()
        hid = self.denc(batch.xc, batch.yc)
        prior = self.lenc(batch.xc, batch.yc)
        if self.training:
            posterior = self.lenc(batch.x, batch.y)
            z = posterior.rsample()
            py = self.dec(torch.cat([hid, z], -1), batch.x)
            outs.recon = py.log_prob(batch.y).sum(-1).mean()
            outs.kld = kl_divergence(posterior, prior).sum(-1).mean()
            outs.loss = outs.kld / batch.x.shape[-2] - outs.recon
        else:
            K = num_samples or 1
            z = prior.rsample([K])
            hid = torch.stack([hid]*K)
            xt = torch.stack([batch.xt]*K)
            yt = torch.stack([batch.yt]*K)
            py = self.dec(torch.cat([hid, z], -1), xt)
            pred_ll = py.log_prob(yt).sum(-1).logsumexp(0) - math.log(K)
            outs.pred_ll = pred_ll.mean()
        return outs

    def predict(self, xc, yc, xt, num_samples=None):
        hid = self.denc(xc, yc)
        prior = self.lenc(xc, yc)
        if num_samples is not None and num_samples > 1:
            z = prior.rsample([num_samples])
            hid = torch.stack([hid]*num_samples)
            xt = torch.stack([xt]*num_samples)
        else:
            z = prior.rsample()
        py = self.dec(torch.cat([hid, z], -1), xt)
        return py.mean, py.scale

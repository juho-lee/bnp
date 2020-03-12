import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence, Dirichlet
from attrdict import AttrDict
import math

from models.anp import DeterministicEncoder, MultiHeadAttention
from models.np import Decoder, LatentEncoder, BootstrapEncoder, \
        sample_bootstrap

class LocalLatentEncoder(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128, num_heads=8):
        super().__init__()
        self.mlp_v = nn.Sequential(
                nn.Linear(dim_x+dim_y, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid))
        self.mlp_qk = nn.Sequential(
                nn.Linear(dim_x, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid))
        self.attn = MultiHeadAttention(dim_hid, dim_hid, dim_hid, 2*dim_lat,
                num_heads=num_heads)
        self.fc_v = nn.Linear(dim_hid, 2*dim_lat)

    def forward(self, xc, yc, xt):
        query, key = self.mlp_qk(xt), self.mlp_qk(xc)
        value = self.mlp_v(torch.cat([xc, yc], -1))
        mu, sigma = self.attn(query, key, value).chunk(2, -1)
        sigma = 0.1 + torch.sigmoid(sigma)
        qc = Normal(mu, sigma)
        mu, sigma = self.fc_v(value.mean(-2)).chunk(2, -1)
        sigma = 0.1 + torch.sigmoid(sigma)
        qz = Normal(mu, sigma)
        return qc, qz

class ANP2(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128, fixed_var=False):
        super().__init__()
        self.denc = DeterministicEncoder(dim_x, dim_y, dim_hid)
        self.lenc = LocalLatentEncoder(dim_x, dim_y, dim_hid, dim_lat)
        self.dec = Decoder(dim_x, dim_y, dim_hid+2*dim_lat, dim_hid, fixed_var)

    def forward(self, batch, num_samples=None):
        outs = AttrDict()
        if self.training:
            hid = self.denc(batch.xc, batch.yc, batch.x)
            cprior, zprior = self.lenc(batch.xc, batch.yc, batch.x)
            cposterior, zposterior = self.lenc(batch.x, batch.y, batch.x)
            c = cposterior.rsample()
            z = zposterior.rsample()
            zc = torch.cat([torch.stack([z]*c.shape[-2], -2), c], -1)
            py = self.dec(torch.cat([hid, zc], -1), batch.x)
            outs.recon = py.log_prob(batch.y).sum(-1).mean()
            outs.ckld = kl_divergence(cposterior, cprior).sum(-1).mean()
            outs.zkld = kl_divergence(zposterior, zprior).sum(-1).mean()
            outs.loss = outs.zkld/batch.x.shape[-2] + outs.ckld - outs.recon
        else:
            K = num_samples or 1
            hid = self.denc(batch.xc, batch.yc, batch.xt)
            cprior, zprior = self.lenc(batch.xc, batch.yc, batch.xt)
            z = zprior.rsample([K])
            c = cprior.rsample([K])
            hid = torch.stack([hid]*K)
            xt = torch.stack([batch.xt]*K)
            yt = torch.stack([batch.yt]*K)
            zc = torch.cat([torch.stack([z]*c.shape[-2], -2), c], -1)
            py = self.dec(torch.cat([hid, zc], -1), xt)
            pred_ll = py.log_prob(yt).sum(-1).logsumexp(0) - math.log(K)
            outs.pred_ll = pred_ll.mean()
        return outs

    def predict(self, xc, yc, xt, num_samples=None):
        hid = self.denc(xc, yc, xt)
        cprior, zprior = self.lenc(xc, yc, xt)
        if num_samples is not None and num_samples > 1:
            z = zprior.rsample([num_samples])
            c = cprior.rsample([num_samples])
            hid = torch.stack([hid]*num_samples)
            xt = torch.stack([xt]*num_samples)
        else:
            z = zprior.rsample()
            c = cprior.rsample()

        zc = torch.cat([torch.stack([z]*c.shape[-2], -2), c], -1)
        py = self.dec(torch.cat([hid, zc], -1), xt)
        return py.mean, py.scale

class LocalBootstrapEncoder(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128, num_heads=8):
        super().__init__()
        self.mlp_v = nn.Sequential(
                nn.Linear(dim_x+dim_y, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid))
        self.mlp_qk = nn.Sequential(
                nn.Linear(dim_x, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid))
        self.attn = MultiHeadAttention(dim_hid, dim_hid, dim_hid, dim_lat,
                num_heads=num_heads)
        self.fc_v = nn.Linear(dim_hid, dim_lat)

    def forward(self, xc, yc, xt, num_samples=None):
        bs_x, bs_y = sample_bootstrap(xc, yc, num_samples=num_samples)
        if num_samples is not None and num_samples > 1:
            xt = torch.stack([xt]*num_samples)
        query, key = self.mlp_qk(xt), self.mlp_qk(bs_x)
        value = self.mlp_v(torch.cat([bs_x, bs_y], -1))
        c = self.attn(query, key, value)
        z = self.fc_v(value.mean(-2))
        return torch.cat([torch.stack([z]*c.shape[-2], -2), c], -1)

class BANP2(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, fixed_var=False):
        super().__init__()
        self.denc = DeterministicEncoder(dim_x, dim_y, dim_hid)
        self.benc = LocalBootstrapEncoder(dim_x, dim_y, dim_hid)
        self.dec = Decoder(dim_x, dim_y, 3*dim_hid, dim_hid, fixed_var)

    def forward(self, batch, num_samples=None):
        outs = AttrDict()
        if self.training:
            hid = self.denc(batch.xc, batch.yc, batch.x)
            zc = self.benc(batch.xc, batch.yc, batch.x)
            py = self.dec(torch.cat([hid, zc], -1), batch.x)
            outs.ll = py.log_prob(batch.y).sum(-1).mean()
            outs.loss = -outs.ll
        else:
            K = num_samples or 1
            hid = self.denc(batch.xc, batch.yc, batch.xt)
            hid = torch.stack([hid]*K)
            zc = self.benc(batch.xc, batch.yc, batch.xt, num_samples=K)
            xt = torch.stack([batch.xt]*K)
            yt = torch.stack([batch.yt]*K)
            py = self.dec(torch.cat([hid, zc], -1), xt)
            pred_ll = py.log_prob(yt).sum(-1).logsumexp(0) - math.log(K)
            outs.pred_ll = pred_ll.mean()
        return outs

    def predict(self, xc, yc, xt, num_samples=None):
        hid = self.denc(xc, yc, xt)
        zc = self.benc(xc, yc, xt, num_samples=num_samples)
        if num_samples is not None and num_samples > 1:
            hid = torch.stack([hid]*num_samples)
            xt = torch.stack([xt]*num_samples)
        py = self.dec(torch.cat([hid, zc], -1), xt)
        return py.mean, py.scale

class BANP3(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128, fixed_var=False):
        super().__init__()
        self.denc = DeterministicEncoder(dim_x, dim_y, dim_hid)
        self.benc = BootstrapEncoder(dim_x, dim_y, dim_hid, dim_lat)
        self.dec = Decoder(dim_x, dim_y, dim_hid+dim_lat, dim_hid, fixed_var)

    def forward(self, batch, num_samples=None):
        outs = AttrDict()
        if self.training:
            hid = self.denc(batch.xc, batch.yc, batch.x)
            z = self.benc(batch.x, batch.y)
            z = torch.stack([z]*hid.shape[-2], -2)
            py = self.dec(torch.cat([hid, z], -1), batch.x)
            outs.ll = py.log_prob(batch.y).sum(-1).mean()
            outs.loss = -outs.ll
        else:
            K = num_samples or 1
            hid = self.denc(batch.xc, batch.yc, batch.xt)
            hid = torch.stack([hid]*K)
            z = self.benc(batch.xc, batch.yc, num_samples=K)
            z = torch.stack([z]*hid.shape[-2], -2)
            xt = torch.stack([batch.xt]*K)
            yt = torch.stack([batch.yt]*K)
            py = self.dec(torch.cat([hid, z], -1), xt)
            pred_ll = py.log_prob(yt).sum(-1).logsumexp(0) - math.log(K)
            outs.pred_ll = pred_ll.mean()
        return outs

    def predict(self, xc, yc, xt, num_samples=None):
        hid = self.denc(xc, yc, xt)
        z = self.benc(xc, yc, num_samples=num_samples)
        if num_samples is not None and num_samples > 1:
            hid = torch.stack([hid]*num_samples)
            xt = torch.stack([xt]*num_samples)
        z = torch.stack([z]*hid.shape[-2], -2)
        py = self.dec(torch.cat([hid, z], -1), xt)
        return py.mean, py.scale

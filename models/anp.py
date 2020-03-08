import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from attrdict import AttrDict
import math

from models.np import Decoder, LatentEncoder, BootstrapEncoder

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_query, dim_key, dim_value, dim_output, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_query, dim_output, bias=False)
        self.fc_k = nn.Linear(dim_key, dim_output, bias=False)
        self.fc_v = nn.Linear(dim_value, dim_output, bias=False)
        self.fc_o = nn.Linear(dim_output, dim_output)

    def forward(self, query, key, value):
        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        query_ = torch.cat(query.chunk(self.num_heads, -1), 0)
        key_ = torch.cat(key.chunk(self.num_heads, -1), 0)
        value_ = torch.cat(value.chunk(self.num_heads, -1), 0)

        A = (query_ @ key_.transpose(-2, -1)) / math.sqrt(query.shape[-1])
        A = torch.softmax(A, -1)
        outs = torch.cat((A @ value_).chunk(self.num_heads, 0), -1)
        outs = query + outs
        outs = outs + F.relu(self.fc_o(outs))
        return outs

class DeterministicEncoder(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, num_heads=8):
        super().__init__()
        self.mlp_v = nn.Sequential(
                nn.Linear(dim_x+dim_y, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid))
        self.mlp_qk = nn.Sequential(
                nn.Linear(dim_x, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid))
        self.attn = MultiHeadAttention(dim_hid, dim_hid, dim_hid, dim_hid,
                num_heads=num_heads)

    def forward(self, xc, yc, xt):
        query, key = self.mlp_qk(xt), self.mlp_qk(xc)
        value = self.mlp_v(torch.cat([xc, yc], -1))
        return self.attn(query, key, value)

class CANP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128):
        super().__init__()
        self.enc = DeterministicEncoder(dim_x, dim_y, dim_hid)
        self.dec = Decoder(dim_x, dim_y, dim_hid, dim_hid)

    def forward(self, batch, num_samples=None):
        outs = AttrDict()
        if self.training:
            hid = self.enc(batch.xc, batch.yc, batch.x)
            py = self.dec(hid, batch.x)
            outs.ll = py.log_prob(batch.y).sum(-1).mean()
            outs.loss = -outs.ll
        else:
            hid = self.enc(batch.xc, batch.yc, batch.xt)
            py = self.dec(hid, batch.xt)
            outs.pred_ll = py.log_prob(batch.yt).sum(-1).mean()
        return outs

    def predict(self, xc, yc, xt, num_samples=None):
        hid = self.enc(xc, yc, xt)
        py = self.dec(hid, xt)
        return py.mean, py.scale

class ANP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128):
        super().__init__()
        self.denc = DeterministicEncoder(dim_x, dim_y, dim_hid)
        self.lenc = LatentEncoder(dim_x, dim_y, dim_hid, dim_lat)
        self.dec = Decoder(dim_x, dim_y, dim_hid+dim_lat, dim_hid)

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

class BANP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128):
        super().__init__()
        self.denc = DeterministicEncoder(dim_x, dim_y, dim_hid)
        self.benc = BootstrapEncoder(dim_x, dim_y, dim_hid, dim_lat)
        self.dec = Decoder(dim_x, dim_y, dim_hid+dim_lat, dim_hid)

    def forward(self, batch, num_samples=None):
        outs = AttrDict()
        if self.training:
            hid = self.denc(batch.xc, batch.yc, batch.x)
            z = self.benc(batch.x, batch.y, r=0.5)
            z = torch.stack([z]*hid.shape[-2], -2)
            py = self.dec(torch.cat([hid, z], -1), batch.x)
            outs.ll = py.log_prob(batch.y).sum(-1).mean()
            outs.loss = -outs.ll
        else:
            K = num_samples or 1
            hid = self.denc(batch.xc, batch.yc, batch.xt)
            hid = torch.stack([hid]*K)
            z = self.benc(batch.xc, batch.yc, num_samples=K, r=0.5)
            z = torch.stack([z]*hid.shape[-2], -2)
            xt = torch.stack([batch.xt]*K)
            yt = torch.stack([batch.yt]*K)
            py = self.dec(torch.cat([hid, z], -1), xt)
            pred_ll = py.log_prob(yt).sum(-1).logsumexp(0) - math.log(K)
            outs.pred_ll = pred_ll.mean()
        return outs

    def predict(self, xc, yc, xt, num_samples=None):
        hid = self.denc(xc, yc, xt)
        z = self.benc(xc, yc, num_samples=num_samples, r=0.5)
        if num_samples is not None and num_samples > 1:
            hid = torch.stack([hid]*num_samples)
            xt = torch.stack([xt]*num_samples)
        z = torch.stack([z]*hid.shape[-2], -2)
        py = self.dec(torch.cat([hid, z], -1), xt)
        return py.mean, py.scale

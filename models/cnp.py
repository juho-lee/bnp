import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence, Dirichlet
import argparse
from attrdict import AttrDict

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
    def __init__(self, dim_x=1, dim_y=1, dim_enc=128, dim_hid=128, fixed_var=False):
        super().__init__()
        self.fixed_var = fixed_var
        self.mlp = nn.Sequential(
                nn.Linear(dim_x+dim_enc, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_y if fixed_var else 2*dim_y))

    def forward(self, encoded, x):
        if encoded.dim() < x.dim():
            encoded = torch.stack([encoded]*x.shape[-2], -2)
        if self.fixed_var:
            mu = self.mlp(torch.cat([encoded, x], -1))
            sigma = 2e-2
            return Normal(mu, sigma)
        else:
            mu, sigma = self.mlp(torch.cat([encoded, x], -1)).chunk(2, -1)
            sigma = 0.1 + 0.9 * F.softplus(sigma)
            return Normal(mu, sigma)

class CNP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, fixed_var=False):
        super().__init__()
        self.enc = DeterministicEncoder(dim_x, dim_y, dim_hid)
        self.dec = Decoder(dim_x, dim_y, dim_hid, dim_hid, fixed_var)

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

def load(args):
    return CNP(fixed_var=args.fixed_var)

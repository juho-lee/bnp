import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from attrdict import AttrDict
import math

from models.cnp import DeterministicEncoder, Decoder

def sample_bootstrap(x, y, num_samples=None, r=1.0):
        B, N, _ = x.shape
        Ns = max(int(r * N), 1)
        if num_samples is None:
            idxs = torch.randint(N, size=[B, Ns, 1]).to(x.device)
            bs_x = torch.gather(x, -2, idxs.repeat(1, 1, x.shape[-1]))
            bs_y = torch.gather(y, -2, idxs.repeat(1, 1, y.shape[-1]))
        else:
            idxs = torch.randint(N, size=[num_samples, B, Ns, 1]).to(x.device)
            bs_x = torch.gather(torch.stack([x]*num_samples), -2,
                    idxs.repeat(1, 1, 1, x.shape[-1]))
            bs_y = torch.gather(torch.stack([y]*num_samples), -2,
                    idxs.repeat(1, 1, 1, y.shape[-1]))
        return bs_x, bs_y

class BootstrapEncoder(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128):
        super().__init__()
        self.pre_mlp = nn.Sequential(
                nn.Linear(dim_x+dim_y, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid))
        self.post_mlp = nn.Sequential(
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_lat))

    def forward(self, x, y, num_samples=None, r=1.2):
        bs_x, bs_y = sample_bootstrap(x, y, num_samples=num_samples, r=r)
        hid = self.pre_mlp(torch.cat([bs_x, bs_y], -1))
        N = hid.shape[-2]
        mask = torch.ones(hid.shape[:-2] + (N//2, 1)).to(hid.device)
        mask = torch.cat([mask,
            torch.bernoulli(0.5*torch.ones(hid.shape[:-2] + (N-N//2, 1))).to(hid.device)],
            -2).detach()
        hid = (mask * hid).sum(-2) / mask.sum(-2)
        return self.post_mlp(hid)

class BNP(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128, fixed_var=False):
        super().__init__()
        self.denc = DeterministicEncoder(dim_x, dim_y, dim_hid)
        self.benc = BootstrapEncoder(dim_x, dim_y, dim_hid, dim_lat)
        self.dec = Decoder(dim_x, dim_y, dim_hid+dim_lat, dim_hid, fixed_var)

    def forward(self, batch, num_samples=None):
        outs = AttrDict()
        hid = self.denc(batch.xc, batch.yc)
        if self.training:
            z = self.benc(batch.x, batch.y)
            py = self.dec(torch.cat([hid, z], -1), batch.x)
            outs.ll = py.log_prob(batch.y).sum(-1).mean()
            outs.loss = -outs.ll
        else:
            K = num_samples or 1
            z = self.benc(batch.xc, batch.yc, num_samples=K)
            hid = torch.stack([hid]*K)
            xt = torch.stack([batch.xt]*K)
            yt = torch.stack([batch.yt]*K)
            py = self.dec(torch.cat([hid, z], -1), xt)
            pred_ll = py.log_prob(yt).sum(-1).logsumexp(0) - math.log(K)
            outs.pred_ll = pred_ll.mean()
        return outs

    def predict(self, xc, yc, xt, num_samples=None):
        hid = self.denc(xc, yc)
        z = self.benc(xc, yc, num_samples=num_samples)
        if num_samples is not None and num_samples > 1:
            hid = torch.stack([hid]*num_samples)
            xt = torch.stack([xt]*num_samples)
        py = self.dec(torch.cat([hid, z], -1), xt)
        return py.mean, py.scale

def load(args):
    return BNP(fixed_var=args.fixed_var)

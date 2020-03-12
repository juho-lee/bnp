import torch
import torch.nn as nn
import torch.nn.functional as F
from attrdict import AttrDict
import math

from models.cnp import Decoder
from models.np import LatentEncoder

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
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, fixed_var=False):
        super().__init__()
        self.enc = DeterministicEncoder(dim_x, dim_y, dim_hid)
        self.dec = Decoder(dim_x, dim_y, dim_hid, dim_hid, fixed_var)

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

def load(args):
    return CANP(fixed_var=args.fixed_var)

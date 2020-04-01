import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math

class Encoder(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=None):
        super().__init__()
        self.use_latent = dim_lat is not None
        self.mlp_pre = nn.Sequential(
                nn.Linear(dim_x+dim_y, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid))
        self.mlp_post = nn.Sequential(
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, 2*dim_lat if self.use_latent else dim_hid))

    def forward(self, x, y):
        hid = self.mlp_pre(torch.cat([x, y], -1)).mean(-2)
        if self.use_latent:
            mu, sigma = self.mlp_post(hid).chunk(2, -1)
            sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
            return Normal(mu, sigma)
        else:
            return self.mlp_post(hid)

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

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_query, dim_key, dim_value, dim_output, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_query, dim_output, bias=False)
        self.fc_k = nn.Linear(dim_key, dim_output, bias=False)
        self.fc_v = nn.Linear(dim_value, dim_output, bias=False)
        self.fc_o = nn.Linear(dim_output, dim_output)

    def forward(self, query, key, value, mask=None):
        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        query_ = torch.cat(query.chunk(self.num_heads, -1), 0)
        key_ = torch.cat(key.chunk(self.num_heads, -1), 0)
        value_ = torch.cat(value.chunk(self.num_heads, -1), 0)

        A_logits = (query_ @ key_.transpose(-2, -1)) / math.sqrt(query.shape[-1])
        if mask is not None:
            mask = torch.stack([mask.squeeze(-1)]*query.shape[-2], -2)
            mask = torch.cat([mask]*self.num_heads, 0)
            A_logits.masked_fill(mask, -float('inf'))
            A = torch.softmax(A_logits, -1)
        else:
            A = torch.softmax(A_logits, -1)

        outs = torch.cat((A @ value_).chunk(self.num_heads, 0), -1)
        outs = query + outs
        outs = outs + F.relu(self.fc_o(outs))
        return outs

class AttEncoder(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=None, num_heads=8):
        super().__init__()
        self.use_latent = dim_lat is not None
        self.mlp_v = nn.Sequential(
                nn.Linear(dim_x+dim_y, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid))
        self.mlp_qk = nn.Sequential(
                nn.Linear(dim_x, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid))
        self.attn = MultiHeadAttention(dim_hid, dim_hid, dim_hid,
                2*dim_lat if self.use_latent else dim_hid,
                num_heads=num_heads)

    def forward(self, xc, yc, xt):
        query, key = self.mlp_qk(xt), self.mlp_qk(xc)
        value = self.mlp_v(torch.cat([xc, yc], -1))
        if self.use_latent:
            mu, sigma = self.attn(query, key, value).chunk(2, -1)
            sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
            return Normal(mu, sigma)
        else:
            return self.attn(query, key, value)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math

class DetEncoder(nn.Module):
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

class LatEncoder(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128, rand=True):
        super().__init__()
        self.rand = rand
        self.mlp_pre = nn.Sequential(
                nn.Linear(dim_x+dim_y, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid))
        self.mlp_post = nn.Sequential(
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, 2*dim_lat if rand else dim_lat))

    def forward(self, x, y):
        hid = self.mlp_pre(torch.cat([x, y], -1)).mean(-2)
        if self.rand:
            mu, sigma = self.mlp_post(hid).chunk(2, -1)
            sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
            return Normal(mu, sigma)
        else:
            return self.mlp_post(hid)

class LatEncoderLarger(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128, rand=True):
        super().__init__()
        self.rand = rand
        self.mlp_pre = nn.Sequential(
                nn.Linear(dim_x+dim_y, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid))
        self.mlp_post = nn.Sequential(
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, 2*dim_lat if rand else dim_lat))

    def forward(self, x, y):
        hid = self.mlp_pre(torch.cat([x, y], -1)).mean(-2)
        if self.rand:
            mu, sigma = self.mlp_post(hid).chunk(2, -1)
            sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
            return Normal(mu, sigma)
        else:
            return self.mlp_post(hid)

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

class AttDetEncoder(nn.Module):
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

class PMA(nn.Module):
    def __init__(self, dim_input, dim_output, num_outputs=1, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.Tensor(num_outputs, dim_output))
        self.attn = MultiHeadAttention(dim_output, dim_input, dim_input, dim_output,
                num_heads=num_heads)
        nn.init.xavier_uniform_(self.query)

    def forward(self, x):
        query = self.query
        query = query[(None,)*(x.dim() - 2)]
        query = query.repeat(x.shape[:-2] + (1,1))
        return self.attn(query, x, x).squeeze(-2)

class LatEncoderPMA(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128, dim_lat=128, rand=True):
        super().__init__()
        self.rand = rand
        self.mlp_pre = nn.Sequential(
                nn.Linear(dim_x+dim_y, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, dim_hid))
        self.pma = PMA(dim_hid, dim_hid)
        self.mlp_post = nn.Sequential(
                nn.Linear(dim_hid, dim_hid), nn.ReLU(True),
                nn.Linear(dim_hid, 2*dim_lat if rand else dim_lat))

    def forward(self, x, y):
        hid = self.pma(self.mlp_pre(torch.cat([x, y], -1)))

        if self.rand:
            mu, sigma = self.mlp_post(hid).chunk(2, -1)
            sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
            return Normal(mu, sigma)
        else:
            return self.mlp_post(hid)

class MultiplicativeInteraction(nn.Module):
    def __init__(self, dim_x, dim_z, dim_output):
        super().__init__()
        self.dim_output = dim_output
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.fc_W = nn.Linear(dim_z, dim_output*dim_x)
        self.fc_b = nn.Linear(dim_z, dim_output)

    def forward(self, x, z):
        # batch_size * dim_output*dim_x
        W = self.fc_W(z)
        b = self.fc_b(z)
        W = W.view(W.shape[:-1] + (self.dim_x, self.dim_output))
        return (x.unsqueeze(-2) @ W).squeeze(-2) + b

if __name__ == '__main__':
    x = torch.rand(2, 100, 4)
    z = torch.rand(2, 100, 15)

    mi = MultiplicativeInteraction(4, 15, 8)
    print(mi(x, z).shape)

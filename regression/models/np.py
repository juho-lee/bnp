import torch
import torch.nn as nn
from torch.distributions import kl_divergence
from attrdict import AttrDict

from utils.misc import stack, logmeanexp
from models.modules import PoolingEncoder, Decoder

class NP(nn.Module):
    def __init__(self,
            dim_x=1,
            dim_y=1,
            dim_hid=128,
            dim_lat=128,
            enc_pre_depth=4,
            enc_post_depth=2,
            dec_depth=3):

        super().__init__()

        self.denc = PoolingEncoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_hid=dim_hid,
                pre_depth=enc_pre_depth,
                post_depth=enc_post_depth)

        self.lenc = PoolingEncoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_hid=dim_hid,
                dim_lat=dim_lat,
                pre_depth=enc_pre_depth,
                post_depth=enc_post_depth)

        self.dec = Decoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_enc=dim_hid+dim_lat,
                dim_hid=dim_hid,
                depth=dec_depth)

    def predict(self, xc, yc, xt, z=None, num_samples=None):
        theta = stack(self.denc(xc, yc), num_samples)
        if z is None:
            pz = self.lenc(xc, yc)
            z = pz.rsample() if num_samples is None \
                    else pz.rsample([num_samples])
        encoded = torch.cat([theta, z], -1)
        encoded = stack(encoded, xt.shape[-2], -2)
        return self.dec(encoded, stack(xt, num_samples))

    def forward(self, batch, num_samples=None):
        outs = AttrDict()
        if self.training:
            pz = self.lenc(batch.xc, batch.yc)
            qz = self.lenc(batch.x, batch.y)
            z = qz.rsample() if num_samples is None else \
                    qz.rsample([num_samples])
            py = self.predict(batch.xc, batch.yc, batch.x,
                    z=z, num_samples=num_samples)
            recon = py.log_prob(stack(batch.y, num_samples)).sum(-1)
            if recon.dim() == 3:
                outs.recon = logmeanexp(recon).mean()
            else:
                outs.recon = recon.mean()
            outs.kld = kl_divergence(qz, pz).sum(-1).mean()
            outs.loss = outs.kld / batch.x.shape[-2] - outs.recon
        else:
            py = self.predict(batch.xc, batch.yc, batch.x, num_samples=num_samples)
            if num_samples is None:
                ll = py.log_prob(batch.y).sum(-1)
            else:
                y = torch.stack([batch.y]*num_samples)
                ll = logmeanexp(py.log_prob(y).sum(-1))
            num_ctx = batch.xc.shape[-2]
            outs.ctx_ll = ll[...,:num_ctx].mean()
            outs.tar_ll = ll[...,num_ctx:].mean()
        return outs

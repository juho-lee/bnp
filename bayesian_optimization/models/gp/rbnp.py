import argparse
import os

import torch
import torch.nn as nn
from attrdict import AttrDict

from utils.paths import results_path
from utils.misc import load_module, stack, logmeanexp
from utils.sampling import sample_with_replacement as SWR, sample_mask

class RBNP(nn.Module):
    def __init__(self, base, dim_x=1, dim_y=1, dim_hid=128):
        super().__init__()
        self.base = base
        self.base.dec.add_ctx(self.base.dim_enc)

    def predict(self, xc, yc, xt, num_samples=None):
        # compute residual
        with torch.no_grad():
            mask = sample_mask(xc.shape[0], xc.shape[1],
                    num_samples=num_samples)
            sxc, syc = stack(xc, num_samples), stack(yc, num_samples)
            encoded = self.base.encode(sxc, syc, sxc, mask=mask)
            py = self.base.dec(encoded, sxc)

            mu, sigma = py.mean, py.scale
            res = SWR((syc - mu)/sigma).detach()
            res = (res - res.mean(-2, keepdim=True))

            bxc = sxc
            byc = mu + sigma * res

        hid1 = self.base.encode(xc, yc, xt, num_samples=num_samples)
        if hid1.dim() == 3:
            hid1 = stack(hid1, num_samples)

        sxt = stack(xt, num_samples)
        hid2 = self.base.encode(bxc, byc, sxt)

        py = self.base.dec(hid1, sxt, ctx=hid2)
        return py

    def forward(self, batch, num_samples=None):
        outs = AttrDict()

        if self.training:
            outs = self.base.forward(batch, num_samples=num_samples)
            base_loss = outs.pop('loss')
            outs.base_loss = base_loss

        py = self.predict(batch.xc, batch.yc, batch.x, num_samples=num_samples)
        ll = py.log_prob(stack(batch.y, num_samples)).sum(-1)
        if ll.dim() == 3:
            ll = logmeanexp(ll)

        if self.training:
            outs.ll = ll.mean()
            outs.loss = -outs.ll + outs.base_loss
        else:
            num_ctx = batch.xc.shape[-2]
            outs.ctx_ll = ll[...,:num_ctx].mean()
            outs.tar_ll = ll[...,num_ctx:].mean()

        return outs

def load(args, cmdline):
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, default='cnp')
    sub_args, cmdline = parser.parse_known_args(cmdline)

    for k, v in sub_args.__dict__.items():
        args.__dict__[k] = v

    base, cmdline = load_module(f'models/gp/{args.base}.py').load(args, cmdline)
    args.root = os.path.join(results_path, 'gp', 'rbnp',
            args.base, args.expid)

    return RBNP(base), cmdline

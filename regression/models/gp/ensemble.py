import argparse
import os
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributions import Normal
from attrdict import AttrDict

from utils.misc import load_module, stack, logmeanexp

class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def predict(self, xc, yc, xt, num_samples=None):
        mus, sigmas = [], []
        for model in self.models:
            py = model.predict(xc, yc, xt, num_samples=num_samples)
            mus.append(py.mean)
            sigmas.append(py.scale)

        if mus[0].dim() == 4:
            mu = torch.cat(mus)
            sigma = torch.cat(sigmas)
        else:
            mu = torch.stack(mus)
            sigma = torch.stack(sigmas)

        return Normal(mu, sigma)

    def forward(self, batch, num_samples=None):
        outs = AttrDict()
        if self.training:
            raise NotImplementedError
        else:
            py = self.predict(batch.xc, batch.yc, batch.x, num_samples=num_samples)
            ll = py.log_prob(stack(batch.y, py.mean.shape[0])).sum(-1)
            ll = logmeanexp(ll)
            num_ctx = batch.xc.shape[-2]
            outs.ctx_ll = ll[...,:num_ctx].mean()
            outs.tar_ll = ll[...,num_ctx:].mean()
        return outs

def load(args, cmdline):
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, default='cnp')
    parser.add_argument('--expids', type=str, nargs='*', default=['run1', 'run2'])
    sub_args, cmdline = parser.parse_known_args(cmdline)

    for k, v in sub_args.__dict__.items():
        args.__dict__[k] = v

    base, cmdline = load_module(f'models/gp/{args.base}.py').load(args, cmdline)
    ROOT = '/nfs/parker/ext01/john/np_new/gp'

    args.root = os.path.join(ROOT, 'ensemble', args.base)

    if not os.path.isdir(args.root):
        os.makedirs(args.root)

    models = []
    for eid in args.expids:
        model = deepcopy(base)
        ckpt = torch.load(os.path.join(ROOT, args.base, eid, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        models.append(model)

    return Ensemble(models), cmdline

import os
from importlib.machinery import SourceFileLoader
import math
import torch
import numpy.random as npr
import random
import time

def gen_load_func(parser, func):
    def load(args, cmdline):
        sub_args, cmdline = parser.parse_known_args(cmdline)
        for k, v in sub_args.__dict__.items():
            args.__dict__[k] = v
        return func(**sub_args.__dict__), cmdline
    return load

def load_module(filename):
    module_name = os.path.splitext(os.path.basename(filename))[0]
    return SourceFileLoader(module_name, filename).load_module()

def logmeanexp(x, dim=0):
    return x.logsumexp(dim) - math.log(x.shape[dim])

def stack(x, num_samples=None, dim=0):
    return x if num_samples is None \
            else torch.stack([x]*num_samples, dim=dim)

# x: K-1 dim tensor
def pad_logits(logits):
    shape = logits.shape
    pad = torch.zeros(shape[:-1] + (1,), device=logits.device)
    return torch.cat([logits, pad], -1)

# y: K dim tensor
def inv_centered_softmax(y, tol=1e-6):
    ty = y + tol
    return ty[...,:-1].log() - ty[...,-1].unsqueeze(-1).log()

def set_seed(seed=None):
    seed = int(time.time()) if seed is None else seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    npr.seed(seed)
    random.seed(seed)

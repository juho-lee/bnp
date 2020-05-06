import argparse
import os
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributions import Normal
from attrdict import AttrDict

from models.gp.ensemble import Ensemble
from utils.misc import load_module, stack, logmeanexp

ROOT = '/nfs/parker/ext01/john/np_new/emnist'

def load(args, cmdline):
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, default='cnp')
    parser.add_argument('--expids', type=str, nargs='*', default=['run1', 'run2'])
    sub_args, cmdline = parser.parse_known_args(cmdline)

    for k, v in sub_args.__dict__.items():
        args.__dict__[k] = v

    base, cmdline = load_module(f'models/emnist/{args.base}.py').load(args, cmdline)

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

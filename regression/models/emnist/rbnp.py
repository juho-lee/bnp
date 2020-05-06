import argparse
import os

import torch
import torch.nn as nn
from attrdict import AttrDict

from utils.paths import results_path
from utils.misc import load_module, stack, logmeanexp
from utils.sampling import sample_with_replacement as SWR, sample_mask

from models.gp.rbnp import RBNP

def load(args, cmdline):
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, default='cnp')
    sub_args, cmdline = parser.parse_known_args(cmdline)

    for k, v in sub_args.__dict__.items():
        args.__dict__[k] = v

    base, cmdline = load_module(f'models/emnist/{args.base}.py').load(args, cmdline)
    args.root = os.path.join(results_path, 'emnist', 'rbnp',
            args.base, args.expid)

    return RBNP(base, dim_x=2), cmdline

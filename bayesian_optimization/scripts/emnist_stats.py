import argparse
import os
import os.path as osp
import numpy as np
from utils.paths import results_path

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='cnp')
parser.add_argument('--expids', type=str, nargs='*',
        default=['run1', 'run2', 'run3', 'run4', 'run5'])
args = parser.parse_args()

root = osp.join(results_path, 'emnist', args.model)
num_trials = len(args.expids)

ctx, tar = np.zeros(num_trials), np.zeros(num_trials)
tn_ctx, tn_tar = np.zeros(num_trials), np.zeros(num_trials)
unseen_ctx, unseen_tar = np.zeros(num_trials), np.zeros(num_trials)

for i, eid in enumerate(args.expids):
    with open(osp.join(root, eid, 'eval.log'), 'r') as f:
        line = f.readline().split()
        for j, token in enumerate(line):
            if token == 'ctx_ll':
                ctx[i] = float(line[j+1])
            if token == 'tar_ll':
                tar[i] = float(line[j+1])

        line = f.readline().split()
        for j, token in enumerate(line):
            if token == 'ctx_ll':
                tn_ctx[i] = float(line[j+1])
            if token == 'tar_ll':
                tn_tar[i] = float(line[j+1])

        line = f.readline().split()
        for j, token in enumerate(line):
            if token == 'ctx_ll':
                unseen_ctx[i] = float(line[j+1])
            if token == 'tar_ll':
                unseen_tar[i] = float(line[j+1])

line = f'& \msd{{{ctx.mean():.3f}}}{{{ctx.std():.3f}}}'
line += f'& \msd{{{tar.mean():.3f}}}{{{tar.std():.3f}}}'
line += f'& \msd{{{unseen_ctx.mean():.3f}}}{{{unseen_ctx.std():.3f}}}'
line += f'& \msd{{{unseen_tar.mean():.3f}}}{{{unseen_tar.std():.3f}}}'
line += f'& \msd{{{tn_ctx.mean():.3f}}}{{{tn_ctx.std():.3f}}}'
line += f'& \msd{{{tn_tar.mean():.3f}}}{{{tn_tar.std():.3f}}} \\\\'
print(line)

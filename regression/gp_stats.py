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

root = osp.join(results_path, 'gp', args.model)
num_trials = len(args.expids)

rbf_ctx, rbf_tar = np.zeros(num_trials), np.zeros(num_trials)
tn_ctx, tn_tar = np.zeros(num_trials), np.zeros(num_trials)
periodic_ctx, periodic_tar = np.zeros(num_trials), np.zeros(num_trials)

for i, eid in enumerate(args.expids):
    with open(osp.join(root, eid, 'eval.log'), 'r') as f:
        line = f.readline().split()
        for j, token in enumerate(line):
            if token == 'ctx_ll':
                rbf_ctx[i] = float(line[j+1])
            if token == 'tar_ll':
                rbf_tar[i] = float(line[j+1])

        line = f.readline().split()
        for j, token in enumerate(line):
            if token == 'ctx_ll':
                tn_ctx[i] = float(line[j+1])
            if token == 'tar_ll':
                tn_tar[i] = float(line[j+1])

        line = f.readline().split()
        for j, token in enumerate(line):
            if token == 'ctx_ll':
                periodic_ctx[i] = float(line[j+1])
            if token == 'tar_ll':
                periodic_tar[i] = float(line[j+1])

print(f'\msd{{{rbf_ctx.mean():.3f}}}{{{rbf_ctx.std():.3f}}}')
print(f'\msd{{{rbf_tar.mean():.3f}}}{{{rbf_tar.std():.3f}}}')
print()
print(f'\msd{{{periodic_ctx.mean():.3f}}}{{{periodic_ctx.std():.3f}}}')
print(f'\msd{{{periodic_tar.mean():.3f}}}{{{periodic_tar.std():.3f}}}')
print()
print(f'\msd{{{tn_ctx.mean():.3f}}}{{{tn_ctx.std():.3f}}}')
print(f'\msd{{{tn_tar.mean():.3f}}}{{{tn_tar.std():.3f}}}')

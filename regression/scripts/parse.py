import argparse
import os
import os.path as osp
import numpy as np
from utils.paths import results_path

def parse(data, model, expids, logfile):
    root = osp.join(results_path, data, model)
    num_trials = len(expids)

    ctx, tar = np.zeros(num_trials), np.zeros(num_trials)
    for i, eid in enumerate(expids):
        with open(osp.join(root, eid, logfile), 'r') as f:
            line = f.readline().split()
            for j, token in enumerate(line):
                if token == 'ctx_ll':
                    ctx[i] = float(line[j+1])
                if token == 'tar_ll':
                    tar[i] = float(line[j+1])

    return ctx, tar

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='gp')
    parser.add_argument('--model', type=str, default='cnp')
    parser.add_argument('--expids', type=str, nargs='*',
            default=['run1', 'run2', 'run3', 'run4', 'run5'])
    args = parser.parse_args()

    if args.data == 'gp':
        ctx, tar = parse('gp', args.model, args.expids, 'eval_rbf.log')
        line = f'{args.model.upper()} & \msd{{{ctx.mean():.3f}}}{{{ctx.std():.3f}}} '
        line += f'& \msd{{{tar.mean():.3f}}}{{{tar.std():.3f}}} '
        ctx, tar = parse('gp', args.model, args.expids, 'eval_matern.log')
        line += f'& \msd{{{ctx.mean():.3f}}}{{{ctx.std():.3f}}} '
        line += f'& \msd{{{tar.mean():.3f}}}{{{tar.std():.3f}}} '
        ctx, tar = parse('gp', args.model, args.expids, 'eval_periodic.log')
        line += f'& \msd{{{ctx.mean():.3f}}}{{{ctx.std():.3f}}} '
        line += f'& \msd{{{tar.mean():.3f}}}{{{tar.std():.3f}}} '
        ctx, tar = parse('gp', args.model, args.expids, 'eval_rbf_tn_-1.0.log')
        line += f'& \msd{{{ctx.mean():.3f}}}{{{ctx.std():.3f}}} '
        line += f'& \msd{{{tar.mean():.3f}}}{{{tar.std():.3f}}} \\\\'
    elif args.data == 'emnist':
        ctx, tar = parse('emnist', args.model, args.expids, 'eval_0-10.log')
        line = f'{args.model.upper()} & \msd{{{ctx.mean():.3f}}}{{{ctx.std():.3f}}} '
        line += f'& \msd{{{tar.mean():.3f}}}{{{tar.std():.3f}}} '
        ctx, tar = parse('emnist', args.model, args.expids, 'eval_10-47.log')
        line += f'& \msd{{{ctx.mean():.3f}}}{{{ctx.std():.3f}}} '
        line += f'& \msd{{{tar.mean():.3f}}}{{{tar.std():.3f}}} '
        ctx, tar = parse('emnist', args.model, args.expids, 'eval_0-10_-1.0.log')
        line += f'& \msd{{{ctx.mean():.3f}}}{{{ctx.std():.3f}}} '
        line += f'& \msd{{{tar.mean():.3f}}}{{{tar.std():.3f}}} \\\\'
    elif args.data == 'celeba':
        ctx, tar = parse('celeba', args.model, args.expids, 'eval.log')
        line = f'{args.model.upper()} & \msd{{{ctx.mean():.3f}}}{{{ctx.std():.3f}}} '
        line += f'& \msd{{{tar.mean():.3f}}}{{{tar.std():.3f}}} '
        ctx, tar = parse('celeba', args.model, args.expids, 'eval_-1.0.log')
        line += f'& \msd{{{ctx.mean():.3f}}}{{{ctx.std():.3f}}} '
        line += f'& \msd{{{tar.mean():.3f}}}{{{tar.std():.3f}}} \\\\'
    elif args.data == 'lotka_volterra':
        ctx, tar = parse('lotka_volterra', args.model, args.expids, 'eval.log')
        line = f'{args.model.upper()} & \msd{{{ctx.mean():.3f}}}{{{ctx.std():.3f}}} '
        line += f'& \msd{{{tar.mean():.3f}}}{{{tar.std():.3f}}} '
        ctx, tar = parse('lotka_volterra', args.model, args.expids, 'hare_lynx.log')
        line += f' & \msd{{{ctx.mean():.3f}}}{{{ctx.std():.3f}}} '
        line += f'& \msd{{{tar.mean():.3f}}}{{{tar.std():.3f}}} \\\\'
    else:
        raise ValueError(f'Invalid data {args.data}')

    print(line)

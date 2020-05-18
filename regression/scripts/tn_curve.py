import argparse
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from utils.paths import results_path
from scripts.parse import parse

def get_results(data, models, expids, logfiles):
    results = {}
    for model in models:
        results[model] = np.zeros((len(expids), len(logfiles)))
        for i, lf in enumerate(logfiles):
            results[model][:,i] = parse(data, model, expids, lf)[1]
    return results

def plot(results):
    markers = ['o', 's', 'd', 'h']
    plt.figure()
    for marker, model in zip(markers, results.keys()):
        mean, std = np.mean(results[model], 0), np.std(results[model], 0)
        plt.errorbar(gam, mean, std,
                label=model.upper(),
                linewidth=2,
                marker=marker,
                markersize=8,
                markerfacecolor='white')
    plt.grid(True, alpha=0.05)
    plt.legend(fontsize=20)
    plt.xlabel(r'$\gamma$', fontsize=20)
    plt.ylabel('target ll', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='gp')
    parser.add_argument('--models', type=str, nargs='*',
            default=['cnp', 'np', 'bnp'])
    parser.add_argument('--expids', type=str, nargs='*',
            default=['run1', 'run2', 'run3', 'run4', 'run5'])
    args = parser.parse_args()

    if args.data == 'gp':
        gam = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16]
        logfiles = [f'eval_rbf_tn_{tn}.log' for tn in gam]
        plot(get_results('gp', args.models, args.expids, logfiles))
    elif args.data == 'emnist':
        gam = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
        logfiles = [f'eval_0-10_{tn}.log' for tn in gam]
        plot(get_results('emnist', args.models, args.expids, logfiles))
    elif args.data == 'celeba':
        gam = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
        logfiles = [f'eval_{tn}.log' for tn in gam]
        plot(get_results('celeba', args.models, args.expids, logfiles))

    plt.show()

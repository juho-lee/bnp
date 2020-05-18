import argparse
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from utils.paths import results_path

models = ['cnp', 'np', 'rbnp/cnp', 'rbnp/np',
        'canp', 'anp', 'rbnp/canp', 'rbnp/anp']
labels = ['CNP', 'NP', 'CNP+BS', 'NP+BS',
        'CANP', 'ANP', 'CANP+BS', 'ANP+BS']
markers = ['o', 's', 'd', 'h']

pps = [1.3, 1.1, 0.9, 0.7, 0.5, 0.3, 0.1]
logfiles = [f'periodic_{pp}.log' for pp in pps]
expids = [f'run{r}' for r in range(1,6)]

results = {}
for model in models:
    results[model] = np.zeros((len(expids), len(logfiles)))
    for i, eid in enumerate(expids):
        for j, lf in enumerate(logfiles):
            filename = osp.join(results_path, 'gp', model, eid, lf)
            with open(filename, 'r') as f:
               tokens = f.readline().split()
               for k, tok in enumerate(tokens):
                   if tok == 'tar_ll':
                       results[model][i,j] = float(tokens[k+1])
                       break

plt.figure()
for marker, label, model in zip(markers, labels[:4], models[:4]):
    mean, std = np.mean(results[model], 0), np.std(results[model], 0)
    plt.errorbar(pps, mean, std,
            label=label,
            marker=marker,
            markersize=10,
            linewidth=2,
            markerfacecolor='white')
plt.gca().invert_xaxis()
plt.grid(True, alpha=0.5)
plt.legend(fontsize=20)
plt.xlabel(r'$p$', fontsize=20)
plt.ylabel('target ll', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.savefig('figures/gp_periodic.pdf', bbox_inches='tight')

plt.figure()
for marker, label, model in zip(markers, labels[4:], models[4:]):
    mean, std = np.mean(results[model], 0), np.std(results[model], 0)
    plt.errorbar(pps, mean, std,
            label=label,
            marker=marker,
            markersize=10,
            linewidth=2,
            markerfacecolor='white')
plt.gca().invert_xaxis()
plt.grid(True, alpha=0.5)
plt.legend(fontsize=20)
plt.xlabel(r'$p$', fontsize=20)
plt.ylabel('target ll', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.savefig('figures/attn_gp_periodic.pdf', bbox_inches='tight')

plt.show()

import matplotlib.pyplot as plt
import numpy as np
from figures.parse import parse

noise_levels = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
num_trials = 3
models = ['bnp', 'banp', 'banp_pp']
labels = ['BNP', 'BANP', 'BANP+global']
markers = ['o', 'd', 'h']

plt.figure()
for i, model in enumerate(models):

    log_diffs = np.zeros((num_trials, len(noise_levels)))

    for j, nl in enumerate(noise_levels):
        results = parse('{}/rbf'.format(model),
                'rbf_htn_{}_crit.log'.format(nl),
                num_trials,
                ['log_diffs'])
        log_diffs[:,j] = results['log_diffs']

    mean, std = log_diffs.mean(0), log_diffs.std(0)
    print(std)

    plt.errorbar(noise_levels, mean, std,
        marker=markers[i],
        linewidth=2,
        markersize=10,
        markerfacecolor='white',
        label=labels[i])

plt.grid(True, alpha=0.5)
plt.legend(fontsize=20)
plt.xlabel('noise level', fontsize=20)
plt.ylabel('discrepancy', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('figures/1d_regression_discrepancy.pdf',
        bbox_inches='tight')

plt.show()

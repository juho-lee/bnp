from collections import OrderedDict
import os
import argparse
import numpy as np
import csv

ROOT = '/nfs/parker/ext01/john/neural_process/1d_regression'

def parse(header, filename, num_trials, metrics):
    path = os.path.join(ROOT, header)
    results = OrderedDict([(key,np.zeros(num_trials)) for key in metrics])
    for i in range(num_trials):
        full_filename = os.path.join(path, 'run{}/{}'.format(i+1, filename))
        with open(full_filename, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            line = next(iter(reader))

            for l, token in enumerate(line):
                if token in metrics:
                    results[token][i] = float(line[l+1][:-1])
    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--header', type=str, default='cnp/rbf')
    parser.add_argument('--filename', type=str, default='rbf_eval.log')
    parser.add_argument('--num_trials', type=int, default=3)
    parser.add_argument('--metrics', type=str, nargs='*', default=['ctx_ll', 'tar_ll'])
    args = parser.parse_args()

    results = parse(args.header, args.filename, args.num_trials, args.metrics)
    for key, vals in results.items():
        print('{} {:.3f} ({:.3f})'.format(key, np.mean(vals), np.std(vals)))

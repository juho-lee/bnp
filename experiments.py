import glob
import os
import pickle
import random
import subprocess

import numpy as np

TIME_BUDGET = 20


def open_pkl(filename):
    with open(filename, "rb") as handle:
        metrics = pickle.load(handle)
    return metrics


def report():
    all_pkl_files = glob.glob("result/*/metrics.pkl")
    all_metrics = [open_pkl(fn) for fn in all_pkl_files]
    exp_filter = lambda d: d["time_budget"] == TIME_BUDGET
    all_metrics = [d for d in all_metrics if exp_filter(d)]
    all_metrics.sort(key=lambda x: x["pred_ll"], reverse=True)

    top_k = 5
    print(f"Showing {top_k} of {len(all_metrics)} total matches")
    for d in all_metrics[:top_k]:
        print(d["pred_ll"], end="")
        print("\t", d["lr"], end="")
        print("\t", d["train_batch_size"])
        # print("\t", os.path.basename(d["root"]))
    print("\n\n")


report()
bs_range = [128 * 2 ** n for n in range(8)]
lr_range = [0.03 * 0.1 ** n for n in np.arange(0, 2, 0.01)]
gpus = [0, 1, 2, 3, 4, 5, 6, 7]
for _ in range(100):
    procs = []
    for gpu in gpus:
        lr = random.choice(lr_range)
        bs = random.choice(bs_range)
        command_list = [
            "python",
            "run.py",
            "--time_budget",
            str(TIME_BUDGET),
            "--lr",
            str(lr),
            "--train_batch_size",
            str(bs),
            "--gpu",
            str(gpu),
        ]
        procs.append(subprocess.Popen(command_list))
    for p in procs:
        p.wait()
    report()

import numpy as np
import matplotlib.pyplot as plt
import os


def get_mean_std(list_):
    mean_ = np.mean(list_, axis=0)
    std_ = np.std(list_, axis=0)
    return mean_, std_

if __name__ == '__main__':
    list_files = os.listdir('./')
    print(list_files)

    list_oracles = [str_file for str_file in list_files if 'oracle_' in str_file and '.npy' in str_file]
    print(list_oracles)

    list_regrets_oracle = []

    for str_file in list_oracles:
        dict_ = np.load(str_file, allow_pickle=True)
        dict_ = dict_[()]

        print(dict_['global'])

        list_regrets_oracle.append(dict_['regrets'])

    mean_oracle, std_oracle = get_mean_std(list_regrets_oracle)

    bx = np.arange(0, mean_oracle.shape[0])
    shade_ = 1.96

    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    ax.plot(bx, mean_oracle, lw=4)
    ax.fill_between(bx,
        mean_oracle - shade_ * std_oracle,
        mean_oracle + shade_ * std_oracle,
        alpha=0.3
    )
    ax.set_xlim([np.min(bx), np.max(bx)])
    ax.grid()
    ax.tick_params(labelsize=20)
    ax.set_xlabel('Iteration', fontsize=24)
    ax.set_ylabel('Instantaneous regret', fontsize=24)

    plt.show()


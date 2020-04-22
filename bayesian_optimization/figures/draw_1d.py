import numpy as np
import matplotlib.pyplot as plt
import os


def get_mean_std(list_):
    mean_ = np.mean(list_, axis=0)
    std_ = np.std(list_, axis=0)
    return mean_, std_

def get_regrets(list_files, str_target):
    list_ = [str_file for str_file in list_files if str_target in str_file and '.npy' in str_file]
    print(list_)

    list_regrets = []

    for str_file in list_:
        dict_ = np.load(str_file, allow_pickle=True)
        dict_ = dict_[()]

        print(dict_['global'])

        list_regrets.append(np.squeeze(dict_['yc']) - dict_['global'])

    return list_regrets

def plot_exp(ax, bx, mean_, std_, shade_, str_label):
    ax.plot(bx, mean_, lw=4, label=str_label)
    ax.fill_between(bx,
        mean_ - shade_ * std_,
        mean_ + shade_ * std_,
        alpha=0.3
    )
   

if __name__ == '__main__':
    list_files = os.listdir('./')
    list_files.sort()
    print(list_files)

    list_regrets_oracle = get_regrets(list_files, 'oracle_')
    mean_oracle, std_oracle = get_mean_std(list_regrets_oracle)

    list_regrets_cnp = get_regrets(list_files, 'cnp_')
    print(list_regrets_cnp)
    mean_cnp, std_cnp = get_mean_std(list_regrets_cnp)

    bx = np.arange(0, mean_oracle.shape[0])
    shade_ = 1.96 * 0.2

    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    plot_exp(ax, bx, mean_oracle, std_oracle, shade_, 'GP (Oracle)')
    plot_exp(ax, bx, mean_cnp, std_cnp, shade_, 'CNP')

    ax.set_xlim([np.min(bx), np.max(bx)])
    ax.tick_params(labelsize=20)
    ax.set_xlabel('Iteration', fontsize=24)
    ax.set_ylabel('Instantaneous regret', fontsize=24)

    ax.grid()
    ax.legend(loc='upper right', fancybox=False, edgecolor='black', fontsize=20)

    plt.show()


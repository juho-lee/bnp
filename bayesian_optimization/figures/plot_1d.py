import numpy as np
import matplotlib.pyplot as plt
import os


def get_mean_std(list_):
    mean_ = np.mean(list_, axis=0)
    std_ = np.std(list_, axis=0)
    return mean_, std_

def get_regrets(list_files, str_target):
    list_dicts = np.load(str_target, allow_pickle=True)
    list_dicts = list_dicts[()]

    list_regrets = []

    for dict_ in list_dicts:
#        print(dict_['global'])

        list_regrets.append((np.squeeze(dict_['yc']) - dict_['global']))
#        list_regrets.append((np.squeeze(dict_['yc']) - dict_['global']) / (np.max(dict_['yc']) - dict_['global']))
#        list_regrets.append(dict_['regrets'])

    regrets = np.array(list_regrets)
    print(np.mean(regrets[:, 0]))
    regrets_cum = np.cumsum(regrets, axis=1)

    return regrets, regrets_cum

def plot_exp(ax, bx, mean_, std_, shade_, str_label):
    ax.plot(bx, mean_, lw=3, label=str_label)
    ax.fill_between(bx,
        mean_ - shade_ * std_,
        mean_ + shade_ * std_,
        alpha=0.3
    )


if __name__ == '__main__':
    list_files = os.listdir('./')
    list_files.sort()
    print(list_files)

    regrets_oracle, regrets_cum_oracle = get_regrets(list_files, 'oracle.npy')
    mean_oracle, std_oracle = get_mean_std(regrets_oracle)

    regrets_np, regrets_cum_np = get_regrets(list_files, 'bo_np.npy')
    mean_np, std_np = get_mean_std(regrets_np)

    regrets_cnp, regrets_cum_cnp = get_regrets(list_files, 'bo_cnp.npy')
    mean_cnp, std_cnp = get_mean_std(regrets_cnp)

    regrets_anp, regrets_cum_anp = get_regrets(list_files, 'bo_anp.npy')
    mean_anp, std_anp = get_mean_std(regrets_anp)

    regrets_canp, regrets_cum_canp = get_regrets(list_files, 'bo_canp.npy')
    mean_canp, std_canp = get_mean_std(regrets_canp)

    regrets_rbnp_np, regrets_cum_rbnp_np = get_regrets(list_files, 'bo_rbnp_np.npy')
    mean_rbnp_np, std_rbnp_np = get_mean_std(regrets_rbnp_np)

    regrets_rbnp_cnp, regrets_cum_rbnp_cnp = get_regrets(list_files, 'bo_rbnp_cnp.npy')
    mean_rbnp_cnp, std_rbnp_cnp = get_mean_std(regrets_rbnp_cnp)

    regrets_rbnp_anp, regrets_cum_rbnp_anp = get_regrets(list_files, 'bo_rbnp_anp.npy')
    mean_rbnp_anp, std_rbnp_anp = get_mean_std(regrets_rbnp_anp)

    regrets_rbnp_canp, regrets_cum_rbnp_canp = get_regrets(list_files, 'bo_rbnp_canp.npy')
    mean_rbnp_canp, std_rbnp_canp = get_mean_std(regrets_rbnp_canp)

    bx = np.arange(0, mean_oracle.shape[0])
    shade_ = 1.96 * 0.1

    # Instantaneous regret
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    plot_exp(ax, bx, mean_oracle, std_oracle, shade_, 'GP (Oracle)')
    plot_exp(ax, bx, mean_np, std_np, shade_, 'NP')
    plot_exp(ax, bx, mean_cnp, std_cnp, shade_, 'CNP')
    plot_exp(ax, bx, mean_anp, std_anp, shade_, 'ANP')
    plot_exp(ax, bx, mean_canp, std_canp, shade_, 'CANP')
    plot_exp(ax, bx, mean_rbnp_np, std_rbnp_np, shade_, 'RBNP/NP')
    plot_exp(ax, bx, mean_rbnp_cnp, std_rbnp_cnp, shade_, 'RBNP/CNP')
    plot_exp(ax, bx, mean_rbnp_anp, std_rbnp_anp, shade_, 'RBNP/ANP')
    plot_exp(ax, bx, mean_rbnp_canp, std_rbnp_canp, shade_, 'RBNP/CANP')

    ax.set_xlim([np.min(bx), np.max(bx)])
    ax.tick_params(labelsize=20)
    ax.set_xlabel('Iteration', fontsize=24)
    ax.set_ylabel('Instantaneous regret', fontsize=24)

    ax.grid()
#    ax.legend(loc='upper right', fancybox=False, edgecolor='black', fontsize=20)
    ax.legend(bbox_to_anchor=(-0.2, 1.02, 1.2, 0.2), loc="lower left",
        mode="expand",
        borderaxespad=0, ncol=3, fancybox=False, fontsize=20)

    plt.savefig('./instantaneous.pdf',
        format='pdf',
        transparent=True,
        bbox_inches='tight'
    )
    plt.show()

    mean_oracle, std_oracle = get_mean_std(regrets_cum_oracle)
    mean_np, std_np = get_mean_std(regrets_cum_np)
    mean_cnp, std_cnp = get_mean_std(regrets_cum_cnp)
    mean_anp, std_anp = get_mean_std(regrets_cum_anp)
    mean_canp, std_canp = get_mean_std(regrets_cum_canp)
    mean_rbnp_np, std_rbnp_np = get_mean_std(regrets_cum_rbnp_np)
    mean_rbnp_cnp, std_rbnp_cnp = get_mean_std(regrets_cum_rbnp_cnp)
    mean_rbnp_anp, std_rbnp_anp = get_mean_std(regrets_cum_rbnp_anp)
    mean_rbnp_canp, std_rbnp_canp = get_mean_std(regrets_cum_rbnp_canp)

    # Cumulative regret
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    plot_exp(ax, bx, mean_oracle, std_oracle, shade_, 'GP (Oracle)')
    plot_exp(ax, bx, mean_np, std_np, shade_, 'NP')
    plot_exp(ax, bx, mean_cnp, std_cnp, shade_, 'NP')
    plot_exp(ax, bx, mean_anp, std_anp, shade_, 'ANP')
    plot_exp(ax, bx, mean_canp, std_canp, shade_, 'CANP')
    plot_exp(ax, bx, mean_rbnp_np, std_rbnp_np, shade_, 'RBNP/NP')
    plot_exp(ax, bx, mean_rbnp_cnp, std_rbnp_cnp, shade_, 'RBNP/CNP')
    plot_exp(ax, bx, mean_rbnp_anp, std_rbnp_anp, shade_, 'RBNP/ANP')
    plot_exp(ax, bx, mean_rbnp_canp, std_rbnp_canp, shade_, 'RBNP/CANP')

    ax.set_xlim([np.min(bx), np.max(bx)])
    ax.tick_params(labelsize=20)
    ax.set_xlabel('Iteration', fontsize=24)
    ax.set_ylabel('Cumulative regret', fontsize=24)

    ax.grid()
#    ax.legend(loc='upper left', fancybox=False, edgecolor='black', fontsize=20)
    ax.legend(bbox_to_anchor=(-0.2, 1.02, 1.2, 0.2), loc="lower left",
        mode="expand",
        borderaxespad=0, ncol=3, fancybox=False, fontsize=20)

    plt.savefig('./cumulative.pdf',
        format='pdf',
        transparent=True,
        bbox_inches='tight'
    )
    plt.show()

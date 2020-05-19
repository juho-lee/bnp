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

    prefix = 'bo_'

    regrets_oracle, regrets_cum_oracle = get_regrets(list_files, '{}oracle.npy'.format(prefix))
    mean_oracle, std_oracle = get_mean_std(regrets_oracle)

    regrets_np, regrets_cum_np = get_regrets(list_files, '{}np.npy'.format(prefix))
    mean_np, std_np = get_mean_std(regrets_np)

    regrets_cnp, regrets_cum_cnp = get_regrets(list_files, '{}cnp.npy'.format(prefix))
    mean_cnp, std_cnp = get_mean_std(regrets_cnp)

    regrets_anp, regrets_cum_anp = get_regrets(list_files, '{}anp.npy'.format(prefix))
    mean_anp, std_anp = get_mean_std(regrets_anp)

    regrets_canp, regrets_cum_canp = get_regrets(list_files, '{}canp.npy'.format(prefix))
    mean_canp, std_canp = get_mean_std(regrets_canp)

    regrets_bnp, regrets_cum_bnp = get_regrets(list_files, '{}bnp.npy'.format(prefix))
    mean_bnp, std_bnp = get_mean_std(regrets_bnp)

    regrets_banp, regrets_cum_banp = get_regrets(list_files, '{}banp.npy'.format(prefix))
    mean_banp, std_banp = get_mean_std(regrets_banp)

    bx = np.arange(0, mean_oracle.shape[0])
    shade_ = 1.96 * 0.1

    # Instantaneous regret
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    plot_exp(ax, bx, mean_oracle, std_oracle, shade_, 'GP (Oracle)')
    plot_exp(ax, bx, mean_np, std_np, shade_, 'NP')
    plot_exp(ax, bx, mean_cnp, std_cnp, shade_, 'CNP')
    plot_exp(ax, bx, mean_bnp, std_bnp, shade_, 'BNP')

    ax.set_xlim([np.min(bx), np.max(bx)])
    ax.tick_params(labelsize=20)
    ax.set_xlabel('Iteration', fontsize=24)
    ax.set_ylabel('Instantaneous regret', fontsize=24)

    ax.grid()
    ax.legend(loc='upper right', fancybox=False, edgecolor='black', fontsize=20)

    plt.savefig('./{}instantaneous_wo_attention.pdf'.format(prefix),
        format='pdf',
        transparent=True,
        bbox_inches='tight'
    )
    plt.show()

    ## 
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    plot_exp(ax, bx, mean_oracle, std_oracle, shade_, 'GP (Oracle)')
    plot_exp(ax, bx, mean_anp, std_anp, shade_, 'ANP')
    plot_exp(ax, bx, mean_canp, std_canp, shade_, 'CANP')
    plot_exp(ax, bx, mean_banp, std_banp, shade_, 'BANP')

    ax.set_xlim([np.min(bx), np.max(bx)])
    ax.tick_params(labelsize=20)
    ax.set_xlabel('Iteration', fontsize=24)
    ax.set_ylabel('Instantaneous regret', fontsize=24)

    ax.grid()
    ax.legend(loc='upper right', fancybox=False, edgecolor='black', fontsize=20)

    plt.savefig('./{}instantaneous_w_attention.pdf'.format(prefix),
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
    mean_bnp, std_bnp = get_mean_std(regrets_cum_bnp)
    mean_banp, std_banp = get_mean_std(regrets_cum_banp)

    # Cumulative regret
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    plot_exp(ax, bx, mean_oracle, std_oracle, shade_, 'GP (Oracle)')
    plot_exp(ax, bx, mean_np, std_np, shade_, 'NP')
    plot_exp(ax, bx, mean_cnp, std_cnp, shade_, 'CNP')
    plot_exp(ax, bx, mean_bnp, std_bnp, shade_, 'BNP')

    ax.set_xlim([np.min(bx), np.max(bx)])
    ax.tick_params(labelsize=20)
    ax.set_xlabel('Iteration', fontsize=24)
    ax.set_ylabel('Cumulative regret', fontsize=24)

    ax.grid()
    ax.legend(loc='upper left', fancybox=False, edgecolor='black', fontsize=20)

    plt.savefig('./{}cumulative_wo_attention.pdf'.format(prefix),
        format='pdf',
        transparent=True,
        bbox_inches='tight'
    )
    plt.show()

    ##
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    plot_exp(ax, bx, mean_oracle, std_oracle, shade_, 'GP (Oracle)')
    plot_exp(ax, bx, mean_anp, std_anp, shade_, 'ANP')
    plot_exp(ax, bx, mean_canp, std_canp, shade_, 'CANP')
    plot_exp(ax, bx, mean_banp, std_banp, shade_, 'BANP')

    ax.set_xlim([np.min(bx), np.max(bx)])
    ax.tick_params(labelsize=20)
    ax.set_xlabel('Iteration', fontsize=24)
    ax.set_ylabel('Cumulative regret', fontsize=24)

    ax.grid()
    ax.legend(loc='upper left', fancybox=False, edgecolor='black', fontsize=20)

    plt.savefig('./{}cumulative_w_attention.pdf'.format(prefix),
        format='pdf',
        transparent=True,
        bbox_inches='tight'
    )
    plt.show()

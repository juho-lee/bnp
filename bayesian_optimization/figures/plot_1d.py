import numpy as np
import matplotlib.pyplot as plt
import os


def get_mean_std(list_):
    mean_ = np.mean(list_, axis=0)
    std_ = np.std(list_, axis=0)
    return mean_, std_

def get_regrets(str_target):
    list_dicts = np.load(str_target, allow_pickle=True)
    list_dicts = list_dicts[()]

    list_regrets = []

    for dict_ in list_dicts:
#        print(dict_['global'])

#        list_regrets.append((np.squeeze(dict_['yc']) - dict_['global']))
        list_regrets.append(dict_['regrets'])

    regrets = np.array(list_regrets)
    print(np.mean(regrets[:, 0]))
    regrets_cum = np.cumsum(regrets, axis=1)

    return regrets, regrets_cum

def plot_exp(ax, bx, mean_, std_, shade_, str_label):
#    ax.errorbar(bx, mean_, yerr=shade_ * std_, lw=3, label=str_label, ls='-')
    ax.plot(bx, mean_, lw=3, label=str_label)
    ax.fill_between(bx,
        mean_ - shade_ * std_,
        mean_ + shade_ * std_,
        alpha=0.2
    )


if __name__ == '__main__':
    list_files = os.listdir('./results')
    list_files.sort()
    print(list_files)

    prefix = 'bo_rbf_noisy_'
    is_oracle = True

    regrets_oracle, regrets_cum_oracle = get_regrets('./results/{}oracle.npy'.format(prefix))
    regrets_np, regrets_cum_np = get_regrets('./results/{}np.npy'.format(prefix))
    regrets_cnp, regrets_cum_cnp = get_regrets('./results/{}cnp.npy'.format(prefix))
    regrets_anp, regrets_cum_anp = get_regrets('./results/{}anp.npy'.format(prefix))
    regrets_canp, regrets_cum_canp = get_regrets('./results/{}canp.npy'.format(prefix))
    regrets_bnp, regrets_cum_bnp = get_regrets('./results/{}bnp.npy'.format(prefix))
    regrets_banp, regrets_cum_banp = get_regrets('./results/{}banp.npy'.format(prefix))

    list_max = []
    list_max.append(np.max(regrets_oracle, axis=1))
    list_max.append(np.max(regrets_np, axis=1))
    list_max.append(np.max(regrets_cnp, axis=1))
    list_max.append(np.max(regrets_anp, axis=1))
    list_max.append(np.max(regrets_canp, axis=1))
    list_max.append(np.max(regrets_bnp, axis=1))
    list_max.append(np.max(regrets_banp, axis=1))
    list_max = np.array(list_max)
    list_max = np.max(list_max, axis=0)[..., np.newaxis]
    print(list_max.shape)
    list_max = np.tile(list_max, (1, regrets_oracle.shape[1]))
    print(list_max.shape)

    regrets_oracle /= list_max
    regrets_np /= list_max
    regrets_cnp /= list_max
    regrets_anp /= list_max
    regrets_canp /= list_max
    regrets_bnp /= list_max
    regrets_banp /= list_max

    regrets_cum_oracle /= list_max
    regrets_cum_np /= list_max
    regrets_cum_cnp /= list_max
    regrets_cum_anp /= list_max
    regrets_cum_canp /= list_max
    regrets_cum_bnp /= list_max
    regrets_cum_banp /= list_max

    mean_oracle, std_oracle = get_mean_std(regrets_oracle)
    mean_np, std_np = get_mean_std(regrets_np)
    mean_cnp, std_cnp = get_mean_std(regrets_cnp)
    mean_anp, std_anp = get_mean_std(regrets_anp)
    mean_canp, std_canp = get_mean_std(regrets_canp)
    mean_bnp, std_bnp = get_mean_std(regrets_bnp)
    mean_banp, std_banp = get_mean_std(regrets_banp)

    bx = np.arange(0, mean_np.shape[0])
    shade_ = 1.96 * 0.08
    print(bx.shape)
    print(mean_np.shape)

    # Instantaneous regret
    fig = plt.figure(figsize=(8, 7))
    ax = fig.gca()

    if is_oracle:
        plot_exp(ax, bx, mean_oracle, std_oracle, shade_, 'GP (Oracle)')
    plot_exp(ax, bx, mean_np, std_np, shade_, 'NP')
    plot_exp(ax, bx, mean_cnp, std_cnp, shade_, 'CNP')
    plot_exp(ax, bx, mean_bnp, std_bnp, shade_, 'BNP')

    ax.set_xlim([np.min(bx), np.max(bx)])
    ax.tick_params(labelsize=20)
    ax.set_xlabel('Iteration', fontsize=24)
    ax.set_ylabel('Minimum simple regret', fontsize=24)

    ##
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    axins = zoomed_inset_axes(ax, 1.5, loc='center')
    if is_oracle:
        plot_exp(axins, bx, mean_oracle, std_oracle, shade_, 'GP (Oracle)')
    plot_exp(axins, bx, mean_np, std_np, shade_, 'NP')
    plot_exp(axins, bx, mean_cnp, std_cnp, shade_, 'CNP')
    plot_exp(axins, bx, mean_bnp, std_bnp, shade_, 'BNP')

    ## limit
    axins.set_xlim([40, 50])
    axins.set_ylim([0.5, 0.62])
    plt.yticks(visible=False)
    plt.xticks(visible=False)

    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    ##

    if prefix == 'bo_rbf_':
        ax.set_title('RBF (NP, CNP, BNP)', fontsize=24)
    elif prefix == 'bo_matern_':
        ax.set_title('Matern (NP, CNP, BNP)', fontsize=24)
    elif prefix == 'bo_periodic_':
        ax.set_title('Periodic (NP, CNP, BNP)', fontsize=24)
    elif prefix == 'bo_rbf_noisy_':
        ax.set_title('RBF+$t$-noise (NP, CNP, BNP)', fontsize=24)
    else:
        pass

    ax.grid()
    ax.legend(loc='upper right', fancybox=False, edgecolor='black', fontsize=20)
#    ax.set_yscale('symlog')

    plt.savefig('./{}instantaneous_wo_attention.pdf'.format(prefix),
        format='pdf',
        transparent=True,
        bbox_inches='tight'
    )
    plt.show()

    ## 
    fig = plt.figure(figsize=(8, 7))
    ax = fig.gca()

    if is_oracle:
        plot_exp(ax, bx, mean_oracle, std_oracle, shade_, 'GP (Oracle)')
    plot_exp(ax, bx, mean_anp, std_anp, shade_, 'ANP')
    plot_exp(ax, bx, mean_canp, std_canp, shade_, 'CANP')
    plot_exp(ax, bx, mean_banp, std_banp, shade_, 'BANP')

    ax.set_xlim([np.min(bx), np.max(bx)])
    ax.tick_params(labelsize=20)
    ax.set_xlabel('Iteration', fontsize=24)
    ax.set_ylabel('Minimum simple regret', fontsize=24)

    ##
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    axins = zoomed_inset_axes(ax, 1.5, loc='center')
    if is_oracle:
        plot_exp(axins, bx, mean_oracle, std_oracle, shade_, 'GP (Oracle)')
    plot_exp(axins, bx, mean_anp, std_np, shade_, 'NP')
    plot_exp(axins, bx, mean_canp, std_cnp, shade_, 'CNP')
    plot_exp(axins, bx, mean_banp, std_bnp, shade_, 'BNP')

    ## limit
    axins.set_xlim([40, 50])
    axins.set_ylim([0.53, 0.61])
    plt.yticks(visible=False)
    plt.xticks(visible=False)

    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    ##

    if prefix == 'bo_rbf_':
        ax.set_title('RBF (ANP, CANP, BANP)', fontsize=24)
    elif prefix == 'bo_matern_':
        ax.set_title('Matern (ANP, CANP, BANP)', fontsize=24)
    elif prefix == 'bo_periodic_':
        ax.set_title('Periodic (ANP, CANP, BANP)', fontsize=24)
    elif prefix == 'bo_rbf_noisy_':
        ax.set_title('RBF+$t$-noise (ANP, CANP, BANP)', fontsize=24)
    else:
        pass

    ax.grid()
    ax.legend(loc='upper right', fancybox=False, edgecolor='black', fontsize=20)
#    ax.set_yscale('symlog')

    plt.savefig('./{}instantaneous_w_attention.pdf'.format(prefix),
        format='pdf',
        transparent=True,
        bbox_inches='tight'
    )
    plt.show()

    shade_ = 1.96 * 0.04

    mean_oracle, std_oracle = get_mean_std(regrets_cum_oracle)
    mean_np, std_np = get_mean_std(regrets_cum_np)
    mean_cnp, std_cnp = get_mean_std(regrets_cum_cnp)
    mean_anp, std_anp = get_mean_std(regrets_cum_anp)
    mean_canp, std_canp = get_mean_std(regrets_cum_canp)
    mean_bnp, std_bnp = get_mean_std(regrets_cum_bnp)
    mean_banp, std_banp = get_mean_std(regrets_cum_banp)

    # Cumulative regret
    fig = plt.figure(figsize=(8, 7))
    ax = fig.gca()

    if is_oracle:
        plot_exp(ax, bx, mean_oracle, std_oracle, shade_, 'GP (Oracle)')
    plot_exp(ax, bx, mean_np, std_np, shade_, 'NP')
    plot_exp(ax, bx, mean_cnp, std_cnp, shade_, 'CNP')
    plot_exp(ax, bx, mean_bnp, std_bnp, shade_, 'BNP')

    ax.set_xlim([np.min(bx), np.max(bx)])
    ax.tick_params(labelsize=20)
    ax.set_xlabel('Iteration', fontsize=24)
    ax.set_ylabel('Cumulative minimum regret', fontsize=24)

    ##
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    axins = zoomed_inset_axes(ax, 1.5, loc='upper left')
    if is_oracle:
        plot_exp(axins, bx, mean_oracle, std_oracle, shade_, 'GP (Oracle)')
    plot_exp(axins, bx, mean_np, std_np, shade_, 'NP')
    plot_exp(axins, bx, mean_cnp, std_cnp, shade_, 'CNP')
    plot_exp(axins, bx, mean_bnp, std_bnp, shade_, 'BNP')

    ## limit
    axins.set_xlim([40, 50])
    axins.set_ylim([26, 33])
    plt.yticks(visible=False)
    plt.xticks(visible=False)

    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    ##

    if prefix == 'bo_rbf_':
        ax.set_title('RBF (NP, CNP, BNP)', fontsize=24)
    elif prefix == 'bo_matern_':
        ax.set_title('Matern (NP, CNP, BNP)', fontsize=24)
    elif prefix == 'bo_periodic_':
        ax.set_title('Periodic (NP, CNP, BNP)', fontsize=24)
    elif prefix == 'bo_rbf_noisy_':
        ax.set_title('RBF+$t$-noise (NP, CNP, BNP)', fontsize=24)
    else:
        pass

    ax.grid()
    ax.legend(loc='lower right', fancybox=False, edgecolor='black', fontsize=20)
#    ax.set_yscale('symlog')

    plt.savefig('./{}cumulative_wo_attention.pdf'.format(prefix),
        format='pdf',
        transparent=True,
        bbox_inches='tight'
    )
    plt.show()

    ##
    fig = plt.figure(figsize=(8, 7))
    ax = fig.gca()

    if is_oracle:
        plot_exp(ax, bx, mean_oracle, std_oracle, shade_, 'GP (Oracle)')
    plot_exp(ax, bx, mean_anp, std_anp, shade_, 'ANP')
    plot_exp(ax, bx, mean_canp, std_canp, shade_, 'CANP')
    plot_exp(ax, bx, mean_banp, std_banp, shade_, 'BANP')

    ax.set_xlim([np.min(bx), np.max(bx)])
    ax.tick_params(labelsize=20)
    ax.set_xlabel('Iteration', fontsize=24)
    ax.set_ylabel('Cumulative minimum regret', fontsize=24)

    ##
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    axins = zoomed_inset_axes(ax, 1.5, loc='upper left')
    if is_oracle:
        plot_exp(axins, bx, mean_oracle, std_oracle, shade_, 'GP (Oracle)')
    plot_exp(axins, bx, mean_anp, std_np, shade_, 'NP')
    plot_exp(axins, bx, mean_canp, std_cnp, shade_, 'CNP')
    plot_exp(axins, bx, mean_banp, std_bnp, shade_, 'BNP')

    ## limit
    axins.set_xlim([40, 50])
    axins.set_ylim([25, 33])
    plt.yticks(visible=False)
    plt.xticks(visible=False)

    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    ##

    if prefix == 'bo_rbf_':
        ax.set_title('RBF (ANP, CANP, BANP)', fontsize=24)
    elif prefix == 'bo_matern_':
        ax.set_title('Matern (ANP, CANP, BANP)', fontsize=24)
    elif prefix == 'bo_periodic_':
        ax.set_title('Periodic (ANP, CANP, BANP)', fontsize=24)
    elif prefix == 'bo_rbf_noisy_':
        ax.set_title('RBF+$t$-noise (ANP, CANP, BANP)', fontsize=24)
    else:
        pass

    ax.grid()
    ax.legend(loc='lower right', fancybox=False, edgecolor='black', fontsize=20)
#    ax.set_yscale('symlog')

    plt.savefig('./{}cumulative_w_attention.pdf'.format(prefix),
        format='pdf',
        transparent=True,
        bbox_inches='tight'
    )
    plt.show()


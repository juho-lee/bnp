import argparse
import torch
import matplotlib.pyplot as plt
import os
import os.path as osp

from utils.paths import datasets_path
from data.lotka_volterra import load_hare_lynx

from models.anp import ANP
from models.banp import BANP
import yaml

from utils.paths import results_path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--expid', type=str, default='run1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hare_lynx', action='store_true')
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # load anp
    with open('configs/lotka_volterra/anp.yaml', 'r') as f:
        config = yaml.safe_load(f)
    anp = ANP(**config).cuda()
    ckpt = torch.load(osp.join(results_path, 'lotka_volterra', 'anp',
        args.expid, 'ckpt.tar'))
    anp.load_state_dict(ckpt['model'])

    # load banp
    with open('configs/lotka_volterra/banp.yaml', 'r') as f:
        config = yaml.safe_load(f)
    banp = BANP(**config).cuda()
    ckpt = torch.load(osp.join(results_path, 'lotka_volterra', 'banp',
        args.expid, 'ckpt.tar'))
    banp.load_state_dict(ckpt['model'])

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.hare_lynx:
        batch = load_hare_lynx(1, 16)[0]
    else:
        eval_data = torch.load(osp.join(datasets_path, 'lotka_volterra', 'eval.tar'))
        bid = torch.randint(len(eval_data), [1]).item()
        batch = eval_data[bid]
    for k, v in batch.items():
        batch[k] = v.cuda()[0:1]

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    def tnp(x):
        return x.squeeze().cpu().data.numpy()

    # standardize
    mux, sigmax = batch.xc.mean(-2, keepdim=True), batch.xc.std(-2, keepdim=True)
    muy, sigmay = batch.yc.mean(-2, keepdim=True), batch.yc.std(-2, keepdim=True)
    batch.x = (batch.x-mux)/(sigmax+1e-5)
    batch.xc = (batch.xc-mux)/(sigmax+1e-5)
    batch.xt = (batch.xt-mux)/(sigmax+1e-5)
    batch.y = (batch.y-muy)/(sigmay+1e-5)
    batch.yc = (batch.yc-muy)/(sigmay+1e-5)
    batch.yt = (batch.yt-muy)/(sigmay+1e-5)

    anp.eval()
    xp = torch.linspace(batch.x[0].min(), batch.x[0].max(), 200).cuda()

    with torch.no_grad():
        outs = anp(batch, num_samples=50)
        print(f'anp {outs.tar_ll.item()}')
        py = anp.predict(batch.xc, batch.yc, xp[None,:,None], num_samples=50)
        mu, sigma = py.mean, py.scale
        bmu = mu.mean(0)
        bvar = sigma.pow(2).mean(0) + mu.pow(2).mean(0) - mu.mean(0).pow(2)
        bsigma = bvar.sqrt()

        rx = tnp(xp*sigmax + mux)
        rmu = tnp(bmu*sigmay + muy)
        rsigma = tnp(bsigma*sigmay)
        upper = rmu[:,0] + rsigma[:,0]
        lower = rmu[:,0] - rsigma[:,0]
        axes[0].plot(rx, rmu[:,0], alpha=0.4, color='blue', zorder=2)
        axes[0].fill_between(rx, lower, upper, alpha=0.2,
                linewidth=0.0, color='blue', zorder=2,
                label='ANP (Predator)')

        axes[1].plot(rx, rmu[:,1], alpha=0.4, color='blue', zorder=2)
        upper = rmu[:,1]+  rsigma[:,1]
        lower = rmu[:,1] - rsigma[:,1]
        axes[1].fill_between(rx, lower, upper, alpha=0.2,
                linewidth=0.0, color='blue', zorder=2,
                label='ANP (Prey)')

    banp.eval()
    with torch.no_grad():
        outs = banp(batch, num_samples=50)
        print(f'banp: {outs.tar_ll.item()}')
        py = banp.predict(batch.xc, batch.yc, xp[None,:,None], num_samples=50)
        mu, sigma = py.mean, py.scale
        bmu = mu.mean(0)
        bvar = sigma.pow(2).mean(0) + mu.pow(2).mean(0) - mu.mean(0).pow(2)
        bsigma = bvar.sqrt()

        rx = tnp(xp*sigmax + mux)
        rmu = tnp(bmu*sigmay + muy)
        rsigma = tnp(bsigma*sigmay)
        upper = rmu[:,0] + rsigma[:,0]
        lower = rmu[:,0] - rsigma[:,0]
        axes[0].plot(rx, rmu[:,0], alpha=0.4, color='orange', zorder=1)
        axes[0].fill_between(rx, lower, upper, alpha=0.2, linewidth=0.0,
                label='BANP (Predator)', color='orange', zorder=1)

        axes[1].plot(rx, rmu[:,1], alpha=0.4, color='orange', zorder=1)
        upper = rmu[:,1]+  rsigma[:,1]
        lower = rmu[:,1] - rsigma[:,1]
        axes[1].fill_between(rx, lower, upper, alpha=0.2,
                linewidth=0.0, color='orange', zorder=1,
                label='BANP (Prey)')

    rxc = tnp(batch.xc[0]*sigmax + mux)
    rxt = tnp(batch.xt[0]*sigmax + mux)
    ryc = tnp(batch.yc[0]*sigmay + muy)
    ryt = tnp(batch.yt[0]*sigmay + muy)

    axes[0].scatter(rxc, ryc[:,0], s=100,
            marker='.', label='Predator context', zorder=3, color='k')
    axes[0].scatter(rxt, ryt[:,0], s=100,
            marker='*', label='Predator target', zorder=3, color='r')
    axes[1].scatter(rxc, ryc[:,1], s=100,
            marker='.', label='Prey context', zorder=3, color='k')
    axes[1].scatter(rxt, ryt[:,1], s=100,
            marker='*', label='Prey target', zorder=3, color='r')

    rx = tnp(batch.x*sigmax + mux)
    axes[0].set_xlim([rx.min(), rx.max()])
    axes[1].set_xlim([rx.min(), rx.max()])

    if args.hare_lynx:
        axes[0].set_xlabel('Year', fontsize=20)
        axes[0].set_ylabel('Population (thousands)', fontsize=20)
        axes[1].set_xlabel('Year', fontsize=20)
        axes[1].set_ylabel('Population (thousands)', fontsize=20)
    else:
        axes[0].set_xlabel('Time', fontsize=20)
        axes[0].set_ylabel('Population', fontsize=20)
        axes[1].set_xlabel('Time', fontsize=20)
        axes[1].set_ylabel('Population', fontsize=20)

    axes[0].tick_params(axis='both', labelsize=15)
    axes[1].tick_params(axis='both', labelsize=15)
    axes[0].legend(fontsize=21, ncol=2,
            columnspacing=0.3,
            handletextpad=0.5)
    axes[1].legend(fontsize=21, ncol=2,
            columnspacing=0.3,
            handletextpad=0.5)
    axes[0].grid('on', alpha=0.5)
    axes[1].grid('on', alpha=0.5)

    if args.save:
        path = 'figures/lotka_volterra'
        if not osp.isdir(path):
            os.makedirs(path)
        figname = f'lotka_volterra_{args.seed}'
        if args.hare_lynx:
            figname += '_hare_lynx'
        figname += '.pdf'
        plt.savefig(osp.join(path, figname), bbox_inches='tight')

    plt.show()

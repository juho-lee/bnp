import torch
import matplotlib.pyplot as plt
import os
import os.path as osp
import argparse

from data.image import img_to_task, coord_to_img
from data.gp import *

from models.anp import ANP
from models.banp import BANP
import yaml

from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.paths import results_path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--expid', type=str, default='run1')
    parser.add_argument('--kernel', type=str, default='rbf')
    parser.add_argument('--t_noise', type=float, default=None)
    parser.add_argument('--num_ctx', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # load anp
    with open('configs/gp/anp.yaml', 'r') as f:
        config = yaml.safe_load(f)
    anp = ANP(**config).cuda()
    ckpt = torch.load(osp.join(results_path, 'gp', 'anp', args.expid, 'ckpt.tar'))
    anp.load_state_dict(ckpt['model'])

    # load banp
    with open('configs/gp/banp.yaml', 'r') as f:
        config = yaml.safe_load(f)
    banp = BANP(**config).cuda()
    ckpt = torch.load(osp.join(results_path, 'gp', 'banp', args.expid, 'ckpt.tar'))
    banp.load_state_dict(ckpt['model'])

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.kernel == 'rbf':
        kernel = RBFKernel()
        title = 'RBF'
    elif args.kernel == 'matern':
        kernel = Matern52Kernel()
        title = 'Matern52'
    elif args.kernel == 'periodic':
        kernel = PeriodicKernel()
        title = 'Periodic'

    if args.t_noise is not None:
        title += r'+$t$-noise'

    sampler = GPSampler(kernel, t_noise=args.t_noise)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    batch = sampler.sample(
            batch_size=1,
            max_num_points=50,
            num_ctx=args.num_ctx,
            device='cuda')

    xp = torch.linspace(-2, 2, 200).cuda()

    plt.figure(figsize=(7, 5))
    plt.title(title, fontsize=20)
    plt.grid(True, alpha=0.5)

    def tnp(x):
        return x.squeeze().cpu().data.numpy()

    anp.eval()
    banp.eval()
    with torch.no_grad():
        outs = anp(batch, num_samples=50)
        py = anp.predict(batch.xc, batch.yc, xp[None,:,None], num_samples=50)
        mu, sigma = py.mean, py.scale
        bmu = mu.mean(0)
        bvar = sigma.pow(2).mean(0) + mu.pow(2).mean(0) - mu.mean(0).pow(2)
        bsigma = bvar.sqrt()

        plt.plot(tnp(xp), tnp(bmu[0]), alpha=0.6, zorder=2, color='blue')
        upper = tnp(bmu[0]+bsigma[0])
        lower = tnp(bmu[0]-bsigma[0])
        plt.fill_between(tnp(xp), lower, upper, alpha=0.2, linewidth=0.0, color='blue',
                label=f'ANP ({outs.tar_ll.item():.3f})', zorder=2)

        outs = banp(batch, num_samples=50)
        py = banp.predict(batch.xc, batch.yc, xp[None,:,None], num_samples=50)
        mu, sigma = py.mean, py.scale
        bmu = mu.mean(0)
        bvar = sigma.pow(2).mean(0) + mu.pow(2).mean(0) - mu.mean(0).pow(2)
        bsigma = bvar.sqrt()

        plt.plot(tnp(xp), tnp(bmu[0]), alpha=0.6, zorder=2, color='orange')
        upper = tnp(bmu[0]+bsigma[0])
        lower = tnp(bmu[0]-bsigma[0])
        plt.fill_between(tnp(xp), lower, upper, alpha=0.2, linewidth=0.0, color='orange',
                label=f'BANP ({outs.tar_ll.item():.3f})', zorder=1)

    plt.scatter(tnp(batch.xc[0]), tnp(batch.yc[0]), s=150,
            marker='.', color='k', label='Context', zorder=3)
    plt.scatter(tnp(batch.xt[0]), tnp(batch.yt[0]), s=150,
            marker='*', color='r', label='Target', zorder=3)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=16, ncol=2,
            columnspacing=0.3,
            handletextpad=0.5)
    plt.xlim([-2, 2])

    if args.save:
        path = 'figures/gp'
        if not osp.isdir(path):
            os.makedirs(path)
        figname = f'gp_{args.seed}_{args.kernel}'
        if args.t_noise is not None:
            figname += f'_{args.t_noise}'
        figname += '.pdf'
        plt.savefig(osp.join(path, figname), bbox_inches='tight')
    plt.show()

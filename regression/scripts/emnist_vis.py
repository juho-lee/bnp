import torch
import matplotlib.pyplot as plt
import os
import os.path as osp
import argparse

from data.image import img_to_task, coord_to_img
from data.emnist import EMNIST

from models.anp import ANP
from models.banp import BANP
import yaml

from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.paths import results_path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--expid', type=str, default='run1')
    parser.add_argument('--class_range', type=int, nargs='*', default=[0,10])
    parser.add_argument('--t_noise', type=float, default=None)
    parser.add_argument('--num_ctx', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_title', action='store_true')
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # load anp
    with open('configs/emnist/anp.yaml', 'r') as f:
        config = yaml.safe_load(f)
    anp = ANP(**config).cuda()
    ckpt = torch.load(osp.join(results_path, 'emnist', 'anp', args.expid, 'ckpt.tar'))
    anp.load_state_dict(ckpt['model'])

    # load banp
    with open('configs/emnist/banp.yaml', 'r') as f:
        config = yaml.safe_load(f)
    banp = BANP(**config).cuda()
    ckpt = torch.load(osp.join(results_path, 'emnist', 'banp', args.expid, 'ckpt.tar'))
    banp.load_state_dict(ckpt['model'])

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    loader = torch.utils.data.DataLoader(
            EMNIST(train=False, class_range=args.class_range),
            batch_size=1, shuffle=True)
    x, _ = next(iter(loader))
    batch = img_to_task(x,
            num_ctx=args.num_ctx,
            target_all=True,
            t_noise=args.t_noise,
            device='cuda')

    fig, axes = plt.subplots(1, 6, figsize=(24,4))
    orig_I = coord_to_img(batch.x, 0.5-batch.y, (1, 28, 28)).clamp(0,1)
    ctx_I = coord_to_img(batch.xc, 0.5-batch.yc, (1, 28, 28)).clamp(0,1)

    anp.eval()
    with torch.no_grad():
        py = anp.predict(batch.xc, batch.yc, batch.x, num_samples=50)
    mu, sigma = py.mean, py.scale
    bmu = mu.mean(0)
    bvar = sigma.pow(2).mean(0) + mu.pow(2).mean(0) - mu.mean(0).pow(2)
    bsigma = bvar.sqrt()
    anp_I = coord_to_img(batch.x, (0.5-bmu).clamp(0,1), (1, 28, 28)).clamp(0,1)
    anp_std = coord_to_img(batch.x, bsigma, (1, 28, 28)).clamp(0,1)

    banp.eval()
    with torch.no_grad():
        py = banp.predict(batch.xc, batch.yc, batch.x, num_samples=50)
    mu, sigma = py.mean, py.scale
    bmu = mu.mean(0)
    bvar = sigma.pow(2).mean(0) + mu.pow(2).mean(0) - mu.mean(0).pow(2)
    bsigma = bvar.sqrt()
    banp_I = coord_to_img(batch.x, (0.5-bmu).clamp(0,1), (1, 28, 28)).clamp(0,1)
    banp_std = coord_to_img(batch.x, bsigma, (1, 28, 28)).clamp(0,1)

    if not args.no_title:
        axes[0].set_title('Original', fontsize=40)
    axes[0].imshow(orig_I[0].cpu().data.numpy().transpose(1,2,0))
    axes[0].axis('off')

    if not args.no_title:
        axes[1].set_title('Context', fontsize=40)
    axes[1].imshow(ctx_I[0].cpu().data.numpy().transpose(1,2,0))
    axes[1].axis('off')

    if not args.no_title:
        axes[2].set_title(r'ANP $\mu$', fontsize=40)
    axes[2].imshow(anp_I[0].cpu().data.numpy().transpose(1,2,0))
    axes[2].axis('off')

    if not args.no_title:
        axes[3].set_title(r'ANP $\sigma$', fontsize=40)
    im = axes[3].imshow(anp_std[0].cpu().data.numpy().transpose(1,2,0).sum(-1),
            cmap='Blues')
    divider = make_axes_locatable(axes[3])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)
    axes[3].axis('off')

    if not args.no_title:
        axes[4].set_title(r'BANP $\mu$', fontsize=40)
    axes[4].imshow(banp_I[0].cpu().data.numpy().transpose(1,2,0))
    axes[4].axis('off')

    if not args.no_title:
        axes[5].set_title(r'BANP $\sigma$', fontsize=40)
    im = axes[5].imshow(banp_std[0].cpu().data.numpy().transpose(1,2,0).sum(-1),
            cmap='Blues')
    divider = make_axes_locatable(axes[5])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)
    axes[5].axis('off')

    plt.tight_layout()

    if args.save:
        figname = f'figures/emnist/emnist_{args.seed}_{args.class_range[0]}-{args.class_range[1]}'
        if args.t_noise is not None:
            figname += f'_{args.t_noise}'
        figname += '.pdf'
        plt.savefig(figname, bbox_inches='tight')
    plt.show()

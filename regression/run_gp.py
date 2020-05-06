import os
import argparse
import json
import time
import matplotlib.pyplot as plt
from attrdict import AttrDict

import torch
import torch.nn as nn

from data.gp import *

from utils.paths import results_path
from utils.misc import load_module
from utils.log import get_logger, RunningAverage

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',
            choices=['train', 'eval', 'plot', 'valid'],
            default='train')
    parser.add_argument('--expid', type=str, default='trial')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='0')

    parser.add_argument('--max_num_points', type=int, default=50)

    parser.add_argument('--model', type=str, default='cnp')
    parser.add_argument('--train_batch_size', type=int, default=100)
    parser.add_argument('--train_num_samples', type=int, default=4)

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=1000)

    parser.add_argument('--eval_seed', type=int, default=42)
    parser.add_argument('--eval_num_batches', type=int, default=1000)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_num_samples', type=int, default=50)

    parser.add_argument('--plot_seed', type=int, default=None)
    parser.add_argument('--plot_batch_size', type=int, default=16)
    parser.add_argument('--plot_num_samples', type=int, default=30)
    parser.add_argument('--plot_num_ctx', type=int, default=10)

    # OOD settings
    parser.add_argument('--ood', action='store_true', default=None)
    parser.add_argument('--t_noise', type=float, default=0.1)

    args, cmdline = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model, cmdline = load_module(f'models/gp/{args.model}.py').load(args, cmdline)

    if not hasattr(args, 'root'):
        args.root = os.path.join(results_path, 'gp', args.model, args.expid)

    if len(cmdline) > 0:
        raise ValueError('unexpected arguments: {}'.format(cmdline))

    model.cuda()

    if args.mode == 'train':
        train(args, model)
    elif args.mode == 'eval':
        eval(args, model)
    elif args.mode == 'plot':
        plot(args, model)

def train(args, model):
    if not os.path.isdir(args.root):
        os.makedirs(args.root)

    with open(os.path.join(args.root, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, sort_keys=True, indent=4)
        print(json.dumps(args.__dict__, sort_keys=True, indent=4))

    sampler = GPSampler(RBFKernel())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_steps)

    if args.resume:
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_step = ckpt.step
    else:
        logfilename = os.path.join(args.root,
                f'train_{time.strftime("%Y%m%d-%H%M")}.log')
        start_step = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    if not args.resume:
        logger.info('Total number of parameters: {}\n'.format(
            sum(p.numel() for p in model.parameters())))

    for step in range(start_step, args.num_steps+1):
        model.train()
        optimizer.zero_grad()
        batch = sampler.sample(
            batch_size=args.train_batch_size,
            max_num_points=args.max_num_points,
            device='cuda')
        outs = model(batch, num_samples=args.train_num_samples)
        outs.loss.backward()
        optimizer.step()
        scheduler.step()

        for key, val in outs.items():
            ravg.update(key, val)

        if step % args.print_freq == 0:
            line = f'{args.model}:{args.expid} step {step} '
            line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
            line += ravg.info()
            logger.info(line)

            if step % args.eval_freq == 0:
                line = eval(args, model)
                logger.info(line + '\n')

            ravg.reset()

        if step % args.save_freq == 0 or step == args.num_steps:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.step = step + 1
            torch.save(ckpt, os.path.join(args.root, 'ckpt.tar'))

    args.mode = 'eval'
    eval(args, model)

def eval(args, model):

    if args.mode == 'eval' and args.model != 'ensemble':
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)

    def _eval(sampler):
        # fix seed to get consistent eval sets
        torch.manual_seed(args.eval_seed)
        torch.cuda.manual_seed(args.eval_seed)

        ravg = RunningAverage()
        model.eval()
        with torch.no_grad():
            for i in range(args.eval_num_batches):
                batch = sampler.sample(
                        batch_size=args.eval_batch_size,
                        max_num_points=args.max_num_points,
                        device='cuda')
                outs = model(batch, num_samples=args.eval_num_samples)
                for key, val in outs.items():
                    ravg.update(key, val)

        torch.manual_seed(time.time())
        torch.cuda.manual_seed(time.time())

        return ravg.info()

    # in distribution
    sampler = GPSampler(RBFKernel())
    line = f'{args.model}:{args.expid} rbf ' + _eval(sampler)
    if args.ood:
        sampler = GPSampler(RBFKernel(), t_noise=args.t_noise)
        line += f'\n{args.model}:{args.expid} rbf tn {args.t_noise} ' + _eval(sampler)
        sampler = GPSampler(PeriodicKernel())
        line += f'\n{args.model}:{args.expid} periodic ' + _eval(sampler)

    if args.mode == 'eval':
        filename = os.path.join(args.root, 'eval.log')
        logger = get_logger(filename, mode='w')
        logger.info(line)

    return line

def plot(args, model):
    ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
    model.load_state_dict(ckpt.model)

    def tnp(x):
        return x.squeeze().cpu().data.numpy()

    if args.plot_seed is not None:
        torch.manual_seed(args.plot_seed)
        torch.cuda.manual_seed(args.plot_seed)

    def _plot(sampler, figtitle):

        batch = sampler.sample(
                batch_size=args.plot_batch_size,
                max_num_points=args.max_num_points,
                num_ctx=args.plot_num_ctx,
                device='cuda')

        xp = torch.linspace(-2, 2, 200).cuda()
        model.eval()

        with torch.no_grad():
            outs = model(batch, num_samples=args.plot_num_samples)
            print(f'ctx_ll {outs.ctx_ll.item():.4f}, tar_ll {outs.tar_ll.item():.4f}')

            py = model.predict(batch.xc, batch.yc,
                    xp[None,:,None].repeat(args.plot_batch_size, 1, 1),
                    num_samples=args.plot_num_samples)
            mu, sigma = py.mean.squeeze(0), py.scale.squeeze(0)

        if args.plot_batch_size > 1:
            nrows = max(args.plot_batch_size//4, 1)
            ncols = min(4, args.plot_batch_size)
            fig, axes = plt.subplots(nrows, ncols,
                    figsize=(5*ncols, 5*nrows),
                    num=figtitle)
            axes = axes.flatten()
        else:
            fig = plt.figure(figsize=(5, 5), num=figtitle)
            axes = [plt.gca()]

        # multi sample
        if mu.dim() == 4:
            for i, ax in enumerate(axes):
                for s in range(mu.shape[0]):
                    ax.plot(tnp(xp), tnp(mu[s][i]), color='steelblue',
                            alpha=max(0.5/args.plot_num_samples, 0.1))
                    ax.fill_between(tnp(xp), tnp(mu[s][i])-tnp(sigma[s][i]),
                            tnp(mu[s][i])+tnp(sigma[s][i]),
                            color='skyblue',
                            alpha=max(0.2/args.plot_num_samples, 0.02),
                            linewidth=0.0)
                ax.scatter(tnp(batch.xc[i]), tnp(batch.yc[i]),
                        color='k', label='context', zorder=mu.shape[0]+1)
                ax.scatter(tnp(batch.xt[i]), tnp(batch.yt[i]),
                        color='orchid', label='target',
                        zorder=mu.shape[0]+1)
                ax.legend()
        else:
            for i, ax in enumerate(axes):
                ax.plot(tnp(xp), tnp(mu[i]), color='steelblue', alpha=0.5)
                ax.fill_between(tnp(xp), tnp(mu[i]-sigma[i]), tnp(mu[i]+sigma[i]),
                        color='skyblue', alpha=0.2, linewidth=0.0)
                ax.scatter(tnp(batch.xc[i]), tnp(batch.yc[i]),
                        color='k', label='context')
                ax.scatter(tnp(batch.xt[i]), tnp(batch.yt[i]),
                        color='orchid', label='target')
                ax.legend()

        plt.tight_layout()

    sampler = GPSampler(RBFKernel())
    _plot(sampler, 'rbf')

    if args.ood:
        sampler = GPSampler(RBFKernel(), t_noise=args.t_noise)
        _plot(sampler, f'rbf tn {args.t_noise}')

        sampler = GPSampler(PeriodicKernel())
        _plot(sampler, 'periodic')

    plt.show()

if __name__ == '__main__':
    main()

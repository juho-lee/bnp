import os
import argparse
import json
import time
import matplotlib.pyplot as plt
from attrdict import AttrDict

import torch
import torch.nn as nn

from models import *
from data.gp import *
from log import get_logger, RunningAverage

ROOT = '/nfs/parker/ext01/john/neural_process'

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--expid', type=str, default='trial')
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--resume', action='store_true', default=False)

    parser.add_argument('--kernel',
            choices=['rbf', 'matern', 'periodic'],
            default='rbf')
    parser.add_argument('--model', type=str,
            choices=['cnp', 'np', 'bnp', 'canp', 'anp', 'banp'],
            default='cnp')
    parser.add_argument('--mode',
            choices=['train', 'eval', 'plot'],
            default='train')

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_steps', type=int, default=50000)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=1000)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_num_points', type=int, default=50)

    parser.add_argument('--eval_seed', type=int, default=42)
    parser.add_argument('--num_eval_batches', type=int, default=1000)
    parser.add_argument('--num_eval_samples', type=int, default=100)

    parser.add_argument('--plot_seed', type=int, default=None)
    parser.add_argument('--plot_batch_size', type=int, default=16)
    parser.add_argument('--num_plot_samples', type=int, default=30)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.root = os.path.join(ROOT, args.model, args.kernel, args.expid) \
            if args.root is None else os.path.join(ROOT, args.root)

    if args.kernel == 'rbf':
        kernel = RBFKernel()
    elif args.kernel == 'matern':
        kernel = Matern52Kernel()
    elif args.kernel == 'periodic':
        kernel = PeriodicKernel()

    sampler = GPSampler(kernel, args.batch_size, args.max_num_points)

    if args.model == 'cnp':
        model = CNP()
    elif args.model == 'np':
        model = NP()
    elif args.model == 'canp':
        model = CANP()
    elif args.model == 'anp':
        model = ANP()
    model.cuda()

    if args.mode == 'train':
        train(args, sampler, model)
    elif args.mode == 'eval':
        eval(args, sampler, model)
    elif args.mode == 'plot':
        plot(args, sampler, model)

def train(args, sampler, model):
    if not os.path.isdir(args.root):
        os.makedirs(args.root)

    with open(os.path.join(args.root, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, sort_keys=True, indent=4)

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
        logfilename = os.path.join(args.root, 'train_{}.log'.format(
            time.strftime('%Y%m%d-%H%M')))
        start_step = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    for step in range(start_step, args.num_steps+1):
        model.train()
        optimizer.zero_grad()
        batch = sampler.sample(device='cuda')
        outs = model(batch)
        outs.loss.backward()
        optimizer.step()
        scheduler.step()

        for key, val in outs.items():
            ravg.update(key, val)

        if step % args.print_freq == 0:
            line = '{}:{}:{} step {} lr {:.3e} '.format(
                    args.model, args.kernel, args.expid, step,
                    optimizer.param_groups[0]['lr'])
            line += ravg.info()
            logger.info(line)

            if step % args.eval_freq == 0:
                logger.info(eval(args, sampler, model) + '\n')

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
    eval(args, sampler, model)

def eval(args, sampler, model):
    if args.mode == 'eval':
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
    ravg = RunningAverage()

    # fix seed to get consistent eval sets
    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    model.eval()
    with torch.no_grad():
        for i in range(args.num_eval_batches):
            batch = sampler.sample(device='cuda')
            outs = model(batch, num_samples=args.num_eval_samples)
            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = '{}:{}:{} eval '.format(args.model, args.kernel, args.expid)
    line += ravg.info()

    if args.mode == 'eval':
        logger = get_logger(os.path.join(args.root, 'eval.log'), mode='w')
        logger.info(line)

    return line

def plot(args, sampler, model):

    def tnp(x):
        return x.squeeze().cpu().data.numpy()

    if args.mode == 'plot':
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)

    if args.plot_seed is not None:
        torch.manual_seed(args.plot_seed)
        torch.cuda.manual_seed(args.plot_seed)

    batch = sampler.sample(batch_size=args.plot_batch_size,
            device='cuda')

    xp = torch.linspace(-2, 2, 200).cuda()
    model.eval()

    with torch.no_grad():
        outs = model(batch, args.num_plot_samples)
        print(outs.pred_ll.item())
        mu, sigma = model.predict(batch.xc, batch.yc,
                xp[None,:,None].repeat(args.plot_batch_size, 1, 1),
                num_samples=args.num_eval_samples)

    if args.plot_batch_size > 1:
        nrows = max(args.plot_batch_size//4, 1)
        ncols = min(4, args.plot_batch_size)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
        axes = axes.flatten()
    else:
        fig = plt.figure(figsize=(5, 5))
        axes = [plt.gca()]

    # multi sample
    if mu.dim() == 4:
        for i, ax in enumerate(axes):
            for s in range(mu.shape[0]):
                ax.plot(tnp(xp), tnp(mu[s][i]), color='steelblue', alpha=0.2)
                ax.fill_between(tnp(xp), tnp(mu[s][i])-tnp(sigma[s][i]),
                        tnp(mu[s][i])+tnp(sigma[s][i]),
                        color='skyblue', alpha=0.02, linewidth=0.0)
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
                    color='skyblue', alpha=0.1, linewidth=0.0)
            ax.scatter(tnp(batch.xc[i]), tnp(batch.yc[i]),
                    color='k', label='context')
            ax.scatter(tnp(batch.xt[i]), tnp(batch.yt[i]),
                    color='orchid', label='target')
            ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

import os
import os.path as osp

import argparse
import yaml

import torch
import torch.nn as nn

import math
import time
import matplotlib.pyplot as plt
from attrdict import AttrDict
from tqdm import tqdm
from copy import deepcopy

from data.gp import *

from utils.misc import load_module, logmeanexp
from utils.paths import results_path, evalsets_path
from utils.log import get_logger, RunningAverage

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',
            choices=['train', 'eval', 'plot', 'ensemble'],
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
    parser.add_argument('--eval_num_batches', type=int, default=3000)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_num_samples', type=int, default=50)
    parser.add_argument('--eval_logfile', type=str, default=None)

    parser.add_argument('--plot_seed', type=int, default=None)
    parser.add_argument('--plot_batch_size', type=int, default=16)
    parser.add_argument('--plot_num_samples', type=int, default=30)
    parser.add_argument('--plot_num_ctx', type=int, default=None)

    # OOD settings
    parser.add_argument('--eval_kernel', type=str, default='rbf')
    parser.add_argument('--t_noise', type=float, default=None)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_cls = getattr(load_module(f'models/{args.model}.py'), args.model.upper())
    with open(f'configs/gp/{args.model}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model = model_cls(**config).cuda()

    args.root = osp.join(results_path, 'gp', args.model, args.expid)

    if args.mode == 'train':
        train(args, model)
    elif args.mode == 'eval':
        eval(args, model)
    elif args.mode == 'plot':
        plot(args, model)
    elif args.mode == 'ensemble':
        ensemble(args, model)

def train(args, model):
    if not osp.isdir(args.root):
        os.makedirs(args.root)

    with open(osp.join(args.root, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

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

def gen_evalset(args):
    if args.eval_kernel == 'rbf':
        kernel = RBFKernel()
    elif args.eval_kernel == 'matern':
        kernel = Matern52Kernel()
    elif args.eval_kernel == 'periodic':
        kernel = PeriodicKernel()
    else:
        raise ValueError(f'Invalid kernel {args.eval_kernel}')

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    sampler = GPSampler(kernel, t_noise=args.t_noise)
    batches = []
    for i in tqdm(range(args.eval_num_batches)):
        batches.append(sampler.sample(
                batch_size=args.eval_batch_size,
                max_num_points=args.max_num_points))

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    path = osp.join(evalsets_path, 'gp')
    if not osp.isdir(path):
        os.makedirs(path)

    filename = f'{args.eval_kernel}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.tar'

    torch.save(batches, osp.join(path, filename))

def eval(args, model):
    if args.mode == 'eval':
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        if args.eval_logfile is None:
            eval_logfile = f'eval_{args.eval_kernel}'
            if args.t_noise is not None:
                eval_logfile += f'_tn_{args.t_noise}'
            eval_logfile += '.log'
        else:
            eval_logfile = args.eval_logfile
        filename = os.path.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    else:
        logger = None

    if args.eval_kernel == 'rbf':
        kernel = RBFKernel()
    elif args.eval_kernel == 'matern':
        kernel = Matern52Kernel()
    elif args.eval_kernel == 'periodic':
        kernel = PeriodicKernel()
    else:
        raise ValueError(f'Invalid kernel {args.eval_kernel}')

    path = osp.join(evalsets_path, 'gp')
    filename = f'{args.eval_kernel}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.tar'
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)

    eval_batches = torch.load(osp.join(path, filename))

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    ravg = RunningAverage()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(eval_batches):
            for key, val in batch.items():
                batch[key] = val.cuda()
            outs = model(batch, num_samples=args.eval_num_samples)
            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = f'{args.model}:{args.expid} {args.eval_kernel} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    line += ravg.info()

    if logger is not None:
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

    kernel = RBFKernel() if args.pp is None else PeriodicKernel(p=args.pp)
    sampler = GPSampler(kernel, t_noise=args.t_noise)

    xp = torch.linspace(-2, 2, 200).cuda()
    batch = sampler.sample(
            batch_size=args.plot_batch_size,
            max_num_points=args.max_num_points,
            num_ctx=args.plot_num_ctx,
            device='cuda')

    model.eval()
    with torch.no_grad():
        outs = model(batch, num_samples=args.eval_num_samples)
        print(f'ctx_ll {outs.ctx_ll.item():.4f}, tar_ll {outs.tar_ll.item():.4f}')

        py = model.predict(batch.xc, batch.yc,
                xp[None,:,None].repeat(args.plot_batch_size, 1, 1),
                num_samples=args.plot_num_samples)
        mu, sigma = py.mean.squeeze(0), py.scale.squeeze(0)

    if args.plot_batch_size > 1:
        nrows = max(args.plot_batch_size//4, 1)
        ncols = min(4, args.plot_batch_size)
        fig, axes = plt.subplots(nrows, ncols,
                figsize=(5*ncols, 5*nrows))
        axes = axes.flatten()
    else:
        fig = plt.figure(figsize=(5, 5))
        axes = [plt.gca()]

    # multi sample
    if mu.dim() == 4:
        #var = sigma.pow(2).mean(0) + mu.pow(2).mean(0) - mu.mean(0).pow(2)
        #sigma = var.sqrt()
        #mu = mu.mean(0)

        for i, ax in enumerate(axes):
            #ax.plot(tnp(xp), tnp(mu[i]), color='steelblue', alpha=0.5)
            #ax.fill_between(tnp(xp), tnp(mu[i]-sigma[i]), tnp(mu[i]+sigma[i]),
            #        color='skyblue', alpha=0.2, linewidth=0.0)
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
    plt.show()

def ensemble(args, model):
    num_runs = 5
    models = []
    for i in range(num_runs):
        model_ = deepcopy(model)
        ckpt = torch.load(osp.join(results_path, 'gp', args.model, f'run{i+1}', 'ckpt.tar'))
        model_.load_state_dict(ckpt['model'])
        model_.cuda()
        model_.eval()
        models.append(model_)

    path = osp.join(evalsets_path, 'gp')
    filename = f'{args.eval_kernel}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.tar'
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)

    eval_batches = torch.load(osp.join(path, filename))

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    ravg = RunningAverage()
    with torch.no_grad():
        for batch in tqdm(eval_batches):
            for key, val in batch.items():
                batch[key] = val.cuda()

            ctx_ll = []
            tar_ll = []
            for model in models:
                outs = model(batch,
                        num_samples=args.eval_num_samples,
                        reduce_ll=False)
                ctx_ll.append(outs.ctx_ll)
                tar_ll.append(outs.tar_ll)

            if ctx_ll[0].dim() == 2:
                ctx_ll = torch.stack(ctx_ll)
                tar_ll = torch.stack(tar_ll)
            else:
                ctx_ll = torch.cat(ctx_ll)
                tar_ll = torch.cat(tar_ll)

            ctx_ll = logmeanexp(ctx_ll).mean()
            tar_ll = logmeanexp(tar_ll).mean()

            ravg.update('ctx_ll', ctx_ll)
            ravg.update('tar_ll', tar_ll)

    filename = f'ensemble_{args.eval_kernel}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.log'
    logger = get_logger(osp.join(results_path, 'gp', args.model, filename), mode='w')
    logger.info(ravg.info())

if __name__ == '__main__':
    main()

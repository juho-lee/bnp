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

from utils.misc import load_module, logmeanexp
from utils.paths import results_path, datasets_path, evalsets_path
from utils.log import get_logger, RunningAverage
from data.lotka_volterra import load_hare_lynx

def standardize(batch):
    with torch.no_grad():
        mu, sigma = batch.xc.mean(-2, keepdim=True), batch.xc.std(-2, keepdim=True)
        sigma[sigma==0] = 1.0
        batch.x = (batch.x - mu) / (sigma + 1e-5)
        batch.xc = (batch.xc - mu) / (sigma + 1e-5)
        batch.xt = (batch.xt - mu) / (sigma + 1e-5)

        mu, sigma = batch.yc.mean(-2, keepdim=True), batch.yc.std(-2, keepdim=True)
        batch.y = (batch.y - mu) / (sigma + 1e-5)
        batch.yc = (batch.yc - mu) / (sigma + 1e-5)
        batch.yt = (batch.yt - mu) / (sigma + 1e-5)
        return batch

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
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=1000)

    parser.add_argument('--eval_seed', type=int, default=42)
    parser.add_argument('--hare_lynx', action='store_true')
    parser.add_argument('--eval_num_samples', type=int, default=50)
    parser.add_argument('--eval_logfile', type=str, default=None)

    parser.add_argument('--plot_seed', type=int, default=None)
    parser.add_argument('--plot_batch_size', type=int, default=16)
    parser.add_argument('--plot_num_samples', type=int, default=30)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_cls = getattr(load_module(f'models/{args.model}.py'), args.model.upper())
    with open(f'configs/lotka_volterra/{args.model}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model = model_cls(**config).cuda()

    args.root = osp.join(results_path, 'lotka_volterra', args.model, args.expid)

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

    train_data = torch.load(osp.join(datasets_path, 'lotka_volterra', 'train.tar'))
    eval_data = torch.load(osp.join(datasets_path, 'lotka_volterra', 'eval.tar'))
    num_steps = len(train_data)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps)

    if args.resume:
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_step = ckpt.step
    else:
        logfilename = osp.join(args.root,
                f'train_{time.strftime("%Y%m%d-%H%M")}.log')
        start_step = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    if not args.resume:
        logger.info('Total number of parameters: {}\n'.format(
            sum(p.numel() for p in model.parameters())))

    for step in range(start_step, num_steps+1):
        model.train()
        optimizer.zero_grad()

        batch = standardize(train_data[step-1])
        for key, val in batch.items():
            batch[key] = val.cuda()

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
                line = eval(args, model, eval_data=eval_data)
                logger.info(line + '\n')

            ravg.reset()

        if step % args.save_freq == 0 or step == num_steps:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.step = step + 1
            torch.save(ckpt, osp.join(args.root, 'ckpt.tar'))

    args.mode = 'eval'
    eval(args, model, eval_data=eval_data)

def eval(args, model, eval_data=None):
    if args.mode == 'eval':
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        if args.eval_logfile is None:
            if args.hare_lynx:
                eval_logfile = 'hare_lynx.log'
            else:
                eval_logfile = 'eval.log'
        else:
            eval_logfile = args.eval_logfile
        filename = osp.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    else:
        logger = None

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    if eval_data is None:
        if args.hare_lynx:
            eval_data = load_hare_lynx(1000, 16)
        else:
            eval_data = torch.load(osp.join(datasets_path, 'lotka_volterra', 'eval.tar'))

    ravg = RunningAverage()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(eval_data):
            batch = standardize(batch)
            for key, val in batch.items():
                batch[key] = val.cuda()
            outs = model(batch, num_samples=args.eval_num_samples)
            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = f'{args.model}:{args.expid} '
    line += ravg.info()

    if logger is not None:
        logger.info(line)

    return line

def ensemble(args, model):
    num_runs = 5
    models = []
    for i in range(num_runs):
        model_ = deepcopy(model)
        ckpt = torch.load(osp.join(results_path, 'lotka_volterra', args.model, f'run{i+1}', 'ckpt.tar'))
        model_.load_state_dict(ckpt['model'])
        model_.cuda()
        model_.eval()
        models.append(model_)

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    if args.hare_lynx:
        eval_data = load_hare_lynx(1000, 16)
    else:
        eval_data = torch.load(osp.join(datasets_path, 'lotka_volterra', 'eval.tar'))

    ravg = RunningAverage()
    with torch.no_grad():
        for batch in tqdm(eval_data):
            batch = standardize(batch)
            for key, val in batch.items():
                batch[key] = val.cuda()

            ctx_ll = []
            tar_ll = []
            for model_ in models:
                outs = model_(batch,
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

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    filename = 'ensemble'
    if args.hare_lynx:
        filename += '_hare_lynx'
    filename += '.log'
    logger = get_logger(osp.join(results_path, 'lotka_volterra', args.model, filename), mode='w')
    logger.info(ravg.info())

def plot(args, model):
    ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
    model.load_state_dict(ckpt.model)

    def tnp(x):
        return x.squeeze().cpu().data.numpy()

    if args.hare_lynx:
        eval_data = load_hare_lynx(1000, 16)
    else:
        eval_data = torch.load(osp.join(datasets_path, 'lotka_volterra', 'eval.tar'))
    bid = torch.randint(len(eval_data), [1]).item()
    batch = standardize(eval_data[bid])

    for k, v in batch.items():
        batch[k] = v.cuda()

    model.eval()
    outs = model(batch, num_samples=args.eval_num_samples)
    print(outs.tar_ll)

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    xp = []
    for b in range(batch.x.shape[0]):
        bx = batch.x[b]
        xp.append(torch.linspace(bx.min()-0.1, bx.max()+0.1, 200))
    xp = torch.stack(xp).unsqueeze(-1).cuda()

    model.eval()
    with torch.no_grad():
        py = model.predict(batch.xc, batch.yc, xp, num_samples=args.plot_num_samples)
    mu, sigma = py.mean, py.scale

    if mu.dim() > 3:
        bmu = mu.mean(0)
        bvar = sigma.pow(2).mean(0) + mu.pow(2).mean(0) - mu.mean(0).pow(2)
        bsigma = bvar.sqrt()
    else:
        bmu = mu
        bsigma = sigma

    for i, ax in enumerate(axes.flatten()):
        ax.plot(tnp(xp[i]), tnp(bmu[i]), alpha=0.5)
        upper = tnp(bmu[i][:,0] + bsigma[i][:,0])
        lower = tnp(bmu[i][:,0] - bsigma[i][:,0])
        ax.fill_between(tnp(xp[i]), lower, upper,
                alpha=0.2, linewidth=0.0, label='predator')

        upper = tnp(bmu[i][:,1] + bsigma[i][:,1])
        lower = tnp(bmu[i][:,1] - bsigma[i][:,1])
        ax.fill_between(tnp(xp[i]), lower, upper,
                alpha=0.2, linewidth=0.0, label='prey')

        ax.scatter(tnp(batch.xc[i]), tnp(batch.yc[i][:,0]), color='k', marker='*')
        ax.scatter(tnp(batch.xc[i]), tnp(batch.yc[i][:,1]), color='k', marker='*')

        ax.scatter(tnp(batch.xt[i]), tnp(batch.yt[i][:,0]), color='orchid', marker='x')
        ax.scatter(tnp(batch.xt[i]), tnp(batch.yt[i][:,1]), color='orchid', marker='x')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

import os
import argparse
import json
import time
import matplotlib.pyplot as plt
from attrdict import AttrDict
import numpy as np
from torch.distributions import MultivariateNormal

import torch
import torch.nn as nn

from utils.misc import load_module
from utils.log import get_logger, RunningAverage

from data.gp_prior import GPPriorSampler
from data.rbf import RBFKernel

from bayeso import gp
from bayeso import covariance

ROOT = './'

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',
            choices=['train', 'eval', 'oracle', 'bo', 'plot', 'valid'],
            default='train')
    parser.add_argument('--expid', '-eid', type=str, default='trial')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='0')

    parser.add_argument('--model', type=str, default='cnp')
    parser.add_argument('--train_data', '-td', type=str, default='rbf')
    parser.add_argument('--train_batch_size', '-tb', type=int, default=100)

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=1000)

    parser.add_argument('--eval_log_file', '-elf', type=str, default=None)
    parser.add_argument('--eval_data', '-ed', type=str, default='rbf')
    parser.add_argument('--eval_batch_size', '-eb', type=int, default=16)
    parser.add_argument('--eval_seed', type=int, default=42)
    parser.add_argument('--num_eval_batches', type=int, default=1000)
    parser.add_argument('--num_eval_samples', type=int, default=100)

    parser.add_argument('--plot_seed', type=int, default=None)
    parser.add_argument('--plot_batch_size', type=int, default=16)
    parser.add_argument('--num_plot_samples', type=int, default=30)

    parser.add_argument('--max_num_points', '-mnp', type=int, default=50)
    parser.add_argument('--heavy_tailed_noise', '-tn', type=float, default=0.0)

    parser.add_argument('--valid_log_file', '-vlf', type=str, default=None)

    args, cmdline = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.root = os.path.join(ROOT, args.model,
            args.train_data, args.expid)

    # load data sampler
    data_file = 'data/{}.py'.format(
            args.train_data if args.mode == 'train' \
                    else args.eval_data)
    sampler, cmdline = load_module(data_file).load(args, cmdline)

    # load model
    model_file = 'models/{}.py'.format(args.model)
    model, cmdline = load_module(model_file).load(args, cmdline)

    if len(cmdline) > 0:
        raise ValueError('unexpected arguments: {}'.format(cmdline))

    model.cuda()

    if args.mode == 'train':
        train(args, sampler, model)
    elif args.mode == 'eval':
        eval(args, sampler, model)
    elif args.mode == 'oracle':
        oracle(args, sampler, model)
    elif args.mode == 'bo':
        bo(args, sampler, model)
    elif args.mode == 'plot':
        plot(args, sampler, model)
    elif args.mode == 'valid':
        if not hasattr(model, 'validate'):
            raise ValueError('invalid model {} to validate'.format(args.model))
        validate(args, sampler, model)

def train(args, sampler, model):
    if not os.path.isdir(args.root):
        os.makedirs(args.root)

    with open(os.path.join(args.root, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, sort_keys=True, indent=4)
        print(json.dumps(args.__dict__, sort_keys=True, indent=4))

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

    if not args.resume:
        logger.info('Total number of parameters: {}\n'.format(
            sum(p.numel() for p in model.parameters())))

    for step in range(start_step, args.num_steps+1):
        model.train()
        optimizer.zero_grad()
        batch = sampler.sample(
            batch_size=args.train_batch_size,
            max_num_points=args.max_num_points,
            heavy_tailed_noise=args.heavy_tailed_noise,
            device='cuda')
        outs = model(batch)
        outs.loss.backward()
        optimizer.step()
        scheduler.step()

        for key, val in outs.items():
            ravg.update(key, val)

        if step % args.print_freq == 0:
            line = '{}:{}:{} step {} lr {:.3e} '.format(
                    args.model, args.train_data, args.expid, step,
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
            batch = sampler.sample(
                    batch_size=args.eval_batch_size,
                    max_num_points=args.max_num_points,
                    heavy_tailed_noise=args.heavy_tailed_noise,
                    device='cuda')
            outs = model(batch, num_samples=args.num_eval_samples)
            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = '{}:{}:{} eval '.format(
            args.model, args.eval_data, args.expid)
    line += ravg.info()

    if args.mode == 'eval':
        if args.eval_log_file is None:
            filename = '{}_'.format(args.eval_data)
            if args.heavy_tailed_noise:
                filename += 'htn_'
            filename += 'eval.log'
        else:
            filename = args.eval_log_file

        filename = os.path.join(args.root, filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logger = get_logger(filename, mode='w')
        logger.info(line)

    return line

def oracle(args, sampler, model):

    def tnp(x):
        return x.squeeze().cpu().data.numpy()

    if args.mode == 'bo':
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)

    # plot_seed is used to fix a random seed.
    if args.plot_seed is not None:
        torch.manual_seed(args.plot_seed)
        torch.cuda.manual_seed(args.plot_seed)

    obj_prior = GPPriorSampler(RBFKernel())

    xp = torch.linspace(-2, 2, 1000).cuda()
    xp_ = xp.unsqueeze(0).unsqueeze(2)

    yp = obj_prior.sample(xp_)
    min_yp = yp.min()
    print(min_yp.cpu().numpy())

    model.eval()

    batch = AttrDict()

    indices_permuted = torch.randperm(yp.shape[1])
    num_init = 5

    batch.x = xp_[:, indices_permuted[:2*num_init], :]
    batch.y = yp[:, indices_permuted[:2*num_init], :]

    batch.xc = xp_[:, indices_permuted[:num_init], :]
    batch.yc = yp[:, indices_permuted[:num_init], :]

    batch.xt = xp_[:, indices_permuted[num_init:2*num_init], :]
    batch.yt = yp[:, indices_permuted[num_init:2*num_init], :]

    X_train = batch.xc.squeeze(0).cpu().numpy()
    Y_train = batch.yc.squeeze(0).cpu().numpy()
    X_test = xp_.squeeze(0).cpu().numpy()

    str_cov = 'se'

    cov_X_X, inv_cov_X_X, hyps = gp.get_optimized_kernel(X_train, Y_train, None, str_cov, is_fixed_noise=True, debug=False)

    prior_mu_train = gp.get_prior_mu(None, X_train)
    prior_mu_test = gp.get_prior_mu(None, X_test)
    cov_X_Xs = covariance.cov_main(str_cov, X_train, X_test, hyps, False)
    cov_Xs_Xs = covariance.cov_main(str_cov, X_test, X_test, hyps, True)
    cov_Xs_Xs = (cov_Xs_Xs + cov_Xs_Xs.T) / 2.0

    mu_ = np.dot(np.dot(cov_X_Xs.T, inv_cov_X_X), Y_train - prior_mu_train) + prior_mu_test
    Sigma_ = cov_Xs_Xs - np.dot(np.dot(cov_X_Xs.T, inv_cov_X_X), cov_X_Xs)
    sigma_ = np.expand_dims(np.sqrt(np.maximum(np.diag(Sigma_), 0.0)), axis=1)

    by = MultivariateNormal(torch.FloatTensor(mu_).squeeze(1), torch.FloatTensor(Sigma_)).rsample().unsqueeze(-1)

    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    ax.plot(tnp(xp), np.squeeze(mu_), color='steelblue', alpha=0.5)
    ax.plot(tnp(xp), tnp(by), color='b', alpha=0.5)
    ax.fill_between(tnp(xp),
        np.squeeze(mu_ - sigma_),
        np.squeeze(mu_ + sigma_),
        color='skyblue', alpha=0.2, linewidth=0.0)
    ax.scatter(tnp(batch.xc), tnp(batch.yc),
        color='k', label='context')
    ax.legend()

    plt.tight_layout()
    plt.show()

def bo(args, sampler, model):

    def tnp(x):
        return x.squeeze().cpu().data.numpy()

    if args.mode == 'bo':
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)

    # plot_seed is used to fix a random seed.
    if args.plot_seed is not None:
        torch.manual_seed(args.plot_seed)
        torch.cuda.manual_seed(args.plot_seed)

    obj_prior = GPPriorSampler(RBFKernel())

    xp = torch.linspace(-2, 2, 1000).cuda()
    xp_ = xp.unsqueeze(0).unsqueeze(2)

    yp = obj_prior.sample(xp_)
    min_yp = yp.min()
    print(min_yp.cpu().numpy())

    model.eval()

    batch = AttrDict()

    indices_permuted = torch.randperm(yp.shape[1])
    num_init = 5

    batch.x = xp_[:, indices_permuted[:2*num_init], :]
    batch.y = yp[:, indices_permuted[:2*num_init], :]

    batch.xc = xp_[:, indices_permuted[:num_init], :]
    batch.yc = yp[:, indices_permuted[:num_init], :]

    batch.xt = xp_[:, indices_permuted[num_init:2*num_init], :]
    batch.yt = yp[:, indices_permuted[num_init:2*num_init], :]

    with torch.no_grad():
        outs = model(batch, num_samples=args.num_plot_samples)
        print('ctx_ll {:.4f} tar ll {:.4f}'.format(
            outs.ctx_ll.item(), outs.tar_ll.item()))
        py = model.predict(batch.xc, batch.yc,
                xp[None,:,None].repeat(1, 1, 1),
                num_samples=args.num_plot_samples)
        mu, sigma = py.mean.squeeze(0), py.scale.squeeze(0)

    fig = plt.figure(figsize=(8, 6))
    axes = [plt.gca()]

    if mu.dim() == 4:
        for i, ax in enumerate(axes):
            for s in range(mu.shape[0]):
                ax.plot(tnp(xp), tnp(mu[s]), color='steelblue',
                        alpha=max(0.5/args.num_plot_samples, 0.1))
                ax.fill_between(tnp(xp),
                        tnp(mu[s]) - tnp(sigma[s]),
                        tnp(mu[s]) + tnp(sigma[s]),
                        color='skyblue',
                        alpha=max(0.2/args.num_plot_samples, 0.02),
                        linewidth=0.0)
            ax.scatter(tnp(batch.xc), tnp(batch.yc),
                    color='k', label='context', zorder=mu.shape[0]+1)
#            ax.scatter(tnp(batch.xt[i]), tnp(batch.yt[i]),
#                    color='orchid', label='target',
#                    zorder=mu.shape[0]+1)
            ax.legend()
    else:
        for i, ax in enumerate(axes):
            ax.plot(tnp(xp), tnp(mu), color='steelblue', alpha=0.5)
            ax.fill_between(tnp(xp), tnp(mu - sigma),
                    tnp(mu + sigma),
                    color='skyblue', alpha=0.2, linewidth=0.0)
            ax.scatter(tnp(batch.xc), tnp(batch.yc),
                    color='k', label='context')
#            ax.scatter(tnp(batch.xt), tnp(batch.yt),
#                    color='orchid', label='target')
            ax.legend()

    plt.tight_layout()
    plt.show()

def plot(args, sampler, model):

    def tnp(x):
        return x.squeeze().cpu().data.numpy()

    if args.mode == 'plot':
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)

    if args.plot_seed is not None:
        torch.manual_seed(args.plot_seed)
        torch.cuda.manual_seed(args.plot_seed)

    batch = sampler.sample(
            batch_size=args.plot_batch_size,
            max_num_points=args.max_num_points,
            heavy_tailed_noise=args.heavy_tailed_noise,
            device='cuda')

    xp = torch.linspace(-2, 2, 200).cuda()
    model.eval()

    with torch.no_grad():
        outs = model(batch, num_samples=args.num_plot_samples)
        print('ctx_ll {:.4f} tar ll {:.4f}'.format(
            outs.ctx_ll.item(), outs.tar_ll.item()))
        py = model.predict(batch.xc, batch.yc,
                xp[None,:,None].repeat(args.plot_batch_size, 1, 1),
                num_samples=args.num_plot_samples)
        mu, sigma = py.mean.squeeze(0), py.scale.squeeze(0)

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
                ax.plot(tnp(xp), tnp(mu[s][i]), color='steelblue',
                        alpha=max(0.5/args.num_plot_samples, 0.1))
                ax.fill_between(tnp(xp), tnp(mu[s][i])-tnp(sigma[s][i]),
                        tnp(mu[s][i])+tnp(sigma[s][i]),
                        color='skyblue',
                        alpha=max(0.2/args.num_plot_samples, 0.02),
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

from tqdm import tqdm

def validate(args, sampler, model):
    ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
    model.load_state_dict(ckpt.model)
    ravg = RunningAverage()

    # fix seed to get consistent eval sets
    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(args.num_eval_batches)):
            batch = sampler.sample(
                    batch_size=args.eval_batch_size,
                    max_num_points=args.max_num_points,
                    heavy_tailed_noise=args.heavy_tailed_noise,
                    device='cuda')
            log_diffs = model.validate(batch, num_samples=args.num_eval_samples)
            ravg.update('log_diffs', log_diffs)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = '{}:{}:{} eval '.format(
            args.model, args.eval_data, args.expid)
    line += ravg.info()

    if args.valid_log_file is None:
        filename = '{}_'.format(args.eval_data)
        if args.heavy_tailed_noise:
            filename += 'htn_'
        filename += 'valid.log'
    else:
        filename = args.valid_log_file

    filename = os.path.join(args.root, filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logger = get_logger(filename, mode='w')
    logger.info(line)

    return line

if __name__ == '__main__':
    main()

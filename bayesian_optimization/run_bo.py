import os
import argparse
import json
import time
import matplotlib.pyplot as plt
from attrdict import AttrDict
from tqdm import tqdm

import numpy as np
import os.path as osp
import yaml

import torch
import torch.nn as nn

from data.gp import *

import bayeso
import bayeso.gp as bayesogp
from bayeso import covariance
from bayeso import acquisition

from utils.paths import results_path
from utils.misc import load_module
from utils.log import get_logger, RunningAverage

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',
            choices=['train', 'eval', 'oracle', 'bo'],
            default='train')
    parser.add_argument('--expid', type=str, default='run1')
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
    parser.add_argument('--ood', action='store_true', default=None)
    parser.add_argument('--t_noise', type=float, default=0.1)
    parser.add_argument('--pp', type=float, default=0.5)

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
    elif args.mode == 'oracle':
        oracle(args, model)
    elif args.mode == 'bo':
        bo(args, model)

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

    if args.mode == 'eval':
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        eval_logfile = 'eval.log' if args.eval_logfile is None else args.eval_logfile
        filename = os.path.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    else:
        logger = None

    def _eval(sampler):
        # fix seed to get consistent eval sets
        torch.manual_seed(args.eval_seed)
        torch.cuda.manual_seed(args.eval_seed)

        ravg = RunningAverage()
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(args.eval_num_batches)):
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
    if logger is not None:
        logger.info(line)

    if args.ood:
        sampler = GPSampler(RBFKernel(), t_noise=args.t_noise)
        line_ = f'{args.model}:{args.expid} rbf tn {args.t_noise} ' + _eval(sampler)
        line += '\n' + line_
        if logger is not None:
            logger.info(line_)

        sampler = GPSampler(PeriodicKernel(p=args.pp))
        line_ = f'{args.model}:{args.expid} periodic ' + _eval(sampler)
        if logger is not None:
            logger.info(line_)
        line += '\n' + line_

    return line

def oracle(args, model):

    seed = 42
    num_all = 100
    num_iter = 50
    num_init = 1
    str_cov = 'se'

    list_dict = []

    for ind_seed in range(1, num_all + 1):
        plot_seed_ = seed * ind_seed

        if plot_seed_ is not None:
            torch.manual_seed(plot_seed_)
            torch.cuda.manual_seed(plot_seed_)

        if os.path.exists('./results/oracle_{}.npy'.format(ind_seed)):
            dict_exp = np.load('./results/oracle_{}.npy'.format(ind_seed), allow_pickle=True)
            dict_exp = dict_exp[()]
            list_dict.append(dict_exp)

            print(dict_exp)
            print(dict_exp['global'])
            print(np.array2string(dict_exp['minima'], separator=','))
            print(np.array2string(dict_exp['regrets'], separator=','))

            continue

        sampler = GPPriorSampler(RBFKernel())

        xp = torch.linspace(-2, 2, 1000).cuda()
        xp_ = xp.unsqueeze(0).unsqueeze(2)

        yp = sampler.sample(xp_)
        min_yp = yp.min()
        print(min_yp.cpu().numpy())

        model.eval()

        batch = AttrDict()
        indices_permuted = torch.randperm(yp.shape[1])

        batch.x = xp_[:, indices_permuted[:2*num_init], :]
        batch.y = yp[:, indices_permuted[:2*num_init], :]

        batch.xc = xp_[:, indices_permuted[:num_init], :]
        batch.yc = yp[:, indices_permuted[:num_init], :]

        batch.xt = xp_[:, indices_permuted[num_init:2*num_init], :]
        batch.yt = yp[:, indices_permuted[num_init:2*num_init], :]

        X_train = batch.xc.squeeze(0).cpu().numpy()
        Y_train = batch.yc.squeeze(0).cpu().numpy()
        X_test = xp_.squeeze(0).cpu().numpy()

        list_min = []
        list_min.append(batch.yc.min().cpu().numpy())

        for ind_iter in range(0, num_iter):
            print('ind_seed {} seed {} iter {}'.format(ind_seed, plot_seed_, ind_iter + 1))

            cov_X_X, inv_cov_X_X, hyps = bayesogp.get_optimized_kernel(X_train, Y_train, None, str_cov, is_fixed_noise=True, debug=False)

            prior_mu_train = bayesogp.get_prior_mu(None, X_train)
            prior_mu_test = bayesogp.get_prior_mu(None, X_test)
            cov_X_Xs = covariance.cov_main(str_cov, X_train, X_test, hyps, False)
            cov_Xs_Xs = covariance.cov_main(str_cov, X_test, X_test, hyps, True)
            cov_Xs_Xs = (cov_Xs_Xs + cov_Xs_Xs.T) / 2.0

            mu_ = np.dot(np.dot(cov_X_Xs.T, inv_cov_X_X), Y_train - prior_mu_train) + prior_mu_test
            Sigma_ = cov_Xs_Xs - np.dot(np.dot(cov_X_Xs.T, inv_cov_X_X), cov_X_Xs)
            sigma_ = np.expand_dims(np.sqrt(np.maximum(np.diag(Sigma_), 0.0)), axis=1)

            acq_vals = -1.0 * acquisition.ei(np.ravel(mu_), np.ravel(sigma_), Y_train)
            ind_ = np.argmin(acq_vals)

            x_new = xp[ind_, None, None, None]
            y_new = yp[:, ind_, None, :]

            batch.x = torch.cat([batch.x, x_new], axis=1)
            batch.y = torch.cat([batch.y, y_new], axis=1)

            batch.xc = torch.cat([batch.xc, x_new], axis=1)
            batch.yc = torch.cat([batch.yc, y_new], axis=1)

            X_train = batch.xc.squeeze(0).cpu().numpy()
            Y_train = batch.yc.squeeze(0).cpu().numpy()

            min_cur = batch.yc.min()
            list_min.append(min_cur.cpu().numpy())

        print(min_yp.cpu().numpy())
        print(np.array2string(np.array(list_min), separator=','))
        print(np.array2string(np.array(list_min) - min_yp.cpu().numpy(), separator=','))

        dict_exp = {
            'seed': plot_seed_,
            'str_cov': str_cov,
            'global': min_yp.cpu().numpy(),
            'minima': np.array(list_min),
            'regrets': np.array(list_min) - min_yp.cpu().numpy(),
            'xc': X_train,
            'yc': Y_train,
            'model': 'oracle',
        }

        np.save('./results/oracle_{}.npy'.format(ind_seed), dict_exp)
        list_dict.append(dict_exp)

    np.save('./figures/oracle.npy', list_dict)

def bo(args, model):

    if args.mode == 'bo':
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)

    plot_seed = 42
    str_cov = 'se'
    num_all = 100
    num_iter = 50
    num_init = 1

    list_dict = []

    for ind_seed in range(1, num_all + 1):
        plot_seed_ = plot_seed * ind_seed

        if plot_seed_ is not None:
            torch.manual_seed(plot_seed_)
            torch.cuda.manual_seed(plot_seed_)

        obj_prior = GPPriorSampler(RBFKernel())

        xp = torch.linspace(-2, 2, 1000).cuda()
        xp_ = xp.unsqueeze(0).unsqueeze(2)

        yp = obj_prior.sample(xp_)
        min_yp = yp.min()
        print(min_yp.cpu().numpy())

        model.eval()

        batch = AttrDict()

        indices_permuted = torch.randperm(yp.shape[1])

        batch.x = xp_[:, indices_permuted[:2*num_init], :]
        batch.y = yp[:, indices_permuted[:2*num_init], :]

        batch.xc = xp_[:, indices_permuted[:num_init], :]
        batch.yc = yp[:, indices_permuted[:num_init], :]

        batch.xt = xp_[:, indices_permuted[num_init:2*num_init], :]
        batch.yt = yp[:, indices_permuted[num_init:2*num_init], :]

        X_train = batch.xc.squeeze(0).cpu().numpy()
        Y_train = batch.yc.squeeze(0).cpu().numpy()
        X_test = xp_.squeeze(0).cpu().numpy()

        list_min = []
        list_min.append(batch.yc.min().cpu().numpy())

        for ind_iter in range(0, num_iter):
            print('ind_seed {} seed {} iter {}'.format(ind_seed, plot_seed_, ind_iter + 1))

            with torch.no_grad():
                outs = model(batch, num_samples=args.plot_num_samples)
                print('ctx_ll {:.4f} tar ll {:.4f}'.format(
                    outs.ctx_ll.item(), outs.tar_ll.item()))
                py = model.predict(batch.xc, batch.yc,
                        xp[None,:,None].repeat(1, 1, 1),
                        num_samples=args.plot_num_samples)
                mu, sigma = py.mean.squeeze(0), py.scale.squeeze(0)

            if mu.dim() == 4:
                print(mu.shape, sigma.shape)
                var = sigma.pow(2).mean(0) + mu.pow(2).mean(0) - mu.mean(0).pow(2)
                sigma = var.sqrt().squeeze(0)
                mu = mu.mean(0).squeeze(0)
                print(mu.shape, sigma.shape)

            mu_ = mu.cpu().numpy()
            sigma_ = sigma.cpu().numpy()

            acq_vals = -1.0 * acquisition.ei(np.ravel(mu_), np.ravel(sigma_), Y_train)
            ind_ = np.argmin(acq_vals)

            x_new = xp[ind_, None, None, None]
            y_new = yp[:, ind_, None, :]

            batch.x = torch.cat([batch.x, x_new], axis=1)
            batch.y = torch.cat([batch.y, y_new], axis=1)

            batch.xc = torch.cat([batch.xc, x_new], axis=1)
            batch.yc = torch.cat([batch.yc, y_new], axis=1)

            X_train = batch.xc.squeeze(0).cpu().numpy()
            Y_train = batch.yc.squeeze(0).cpu().numpy()

            min_cur = batch.yc.min()
            list_min.append(min_cur.cpu().numpy())

        print(min_yp.cpu().numpy())
        print(np.array2string(np.array(list_min), separator=','))
        print(np.array2string(np.array(list_min) - min_yp.cpu().numpy(), separator=','))

        dict_exp = {
            'seed': plot_seed_,
            'global': min_yp.cpu().numpy(),
            'minima': np.array(list_min),
            'regrets': np.array(list_min) - min_yp.cpu().numpy(),
            'xc': X_train,
            'yc': Y_train,
            'model': args.model,
            'cov': str_cov,
        }

        list_dict.append(dict_exp)

    np.save('./figures/{}.npy'.format(args.model), list_dict)

if __name__ == '__main__':
    main()

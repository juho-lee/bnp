import os
import argparse
import json
import time
from attrdict import AttrDict
import numpy as np
from torch.distributions import MultivariateNormal

import torch
import torch.nn as nn

from utils.misc import load_module
from utils.log import get_logger, RunningAverage

from data.gp_prior import GPPriorSampler
from data.rbf import RBFKernel

ROOT = './'

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',
            choices=['train', 'eval', 'oracle', 'bo', 'valid'],
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

        from sklearn.gaussian_process import GaussianProcessRegressor

        for ind_iter in range(0, num_iter):
            print('ind_seed {} seed {} iter {}'.format(ind_seed, plot_seed_, ind_iter + 1))

            gpr = GaussianProcessRegressor()
            gpr.fit(X_train, Y_train)
            mu_, Sigma_ = gpr.predict(X_test, return_cov=True)
            Sigma_ += 1e-5 * np.eye(Sigma_.shape[0])

            by = MultivariateNormal(torch.FloatTensor(mu_).squeeze(1), torch.FloatTensor(Sigma_)).rsample().unsqueeze(-1)
            by_ = tnp(by)
            ind_ = np.argmin(by_)

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
            'mininums': np.array(list_min),
            'regrets': np.array(list_min) - min_yp.cpu().numpy(),
            'xc': X_train,
            'yc': Y_train,
            'model': 'oracle',
            'cov': str_cov,
        }

        np.save('./figures/oracle_{}.npy'.format(plot_seed_), dict_exp)
        list_dict.append(dict_exp)

    np.save('./figures/oracle.npy', list_dict)

def bo(args, sampler, model):

    def tnp(x):
        return x.squeeze().cpu().data.numpy()

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
                outs = model(batch, num_samples=args.num_plot_samples)
                print('ctx_ll {:.4f} tar ll {:.4f}'.format(
                    outs.ctx_ll.item(), outs.tar_ll.item()))
                py = model.predict(batch.xc, batch.yc,
                        xp[None,:,None].repeat(1, 1, 1),
                        num_samples=args.num_plot_samples)
                mu, sigma = py.mean.squeeze(0), py.scale.squeeze(0)

            if mu.dim() == 4:
                mu_ = torch.mean(mu, axis=0).squeeze(0)
                _mu_ = (mu - mu_.unsqueeze(0)).squeeze(1).squeeze(2)
                Sigma = torch.sum(torch.einsum('ij,ik->ijk', _mu_, _mu_), axis=0) / (mu.shape[0] - 1) + 1e-5 * torch.eye(mu_.shape[0]).cuda()

#            Sigma = sigma**2 * torch.eye(sigma.shape[0]).cuda()

            by = MultivariateNormal(mu_.squeeze(1), Sigma).rsample().unsqueeze(-1)
            by_ = tnp(by)
            ind_ = np.argmin(by_)

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
            'mininums': np.array(list_min),
            'regrets': np.array(list_min) - min_yp.cpu().numpy(),
            'xc': X_train,
            'yc': Y_train,
            'model': args.model,
            'cov': str_cov,
        }

        np.save('./figures/{}_{}.npy'.format(args.model, plot_seed_), dict_exp)
        list_dict.append(dict_exp)

    np.save('./figures/{}.npy'.format(args.model), list_dict)

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

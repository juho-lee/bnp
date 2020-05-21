import os
import argparse
from attrdict import AttrDict

import numpy as np
import os.path as osp
import yaml

import torch
from data.gp import *

import bayeso
import bayeso.gp as bayesogp
from bayeso import covariance
from bayeso import acquisition

from utils.paths import results_path
from utils.misc import load_module

def get_str_file(path_, str_kernel, str_model, noise, seed=None):
    if noise is not None:
        str_all = 'bo_{}_{}_{}'.format(str_kernel, 'noisy', str_model)
    else:
        str_all = 'bo_{}_{}'.format(str_kernel, str_model)

    if seed is not None:
        str_all += '_' + str(seed) + '.npy'
    else:
        str_all += '.npy'

    return osp.join(path_, str_all)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',
            choices=['oracle', 'bo'],
            default='bo')
    parser.add_argument('--expid', type=str, default='run1')
    parser.add_argument('--gpu', type=str, default='0')

    parser.add_argument('--model', type=str, default='cnp')

    parser.add_argument('--bo_num_samples', type=int, default=200)
    parser.add_argument('--bo_num_init', type=int, default=1)
    parser.add_argument('--bo_kernel', type=str, default='periodic')
    parser.add_argument('--t_noise', type=float, default=None)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_cls = getattr(load_module(f'models/{args.model}.py'), args.model.upper())
    with open(f'configs/gp/{args.model}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model = model_cls(**config).cuda()

    args.root = osp.join(results_path, 'gp', args.model, args.expid)

    if args.mode == 'oracle':
        oracle(args, model)
    elif args.mode == 'bo':
        bo(args, model)

def oracle(args, model):
    seed = 42
    num_all = 100
    num_iter = 50
    num_init = args.bo_num_init
    str_cov = 'se'

    list_dict = []

    if args.bo_kernel == 'rbf':
        kernel = RBFKernel()
    elif args.bo_kernel == 'matern':
        kernel = Matern52Kernel()
    elif args.bo_kernel == 'periodic':
        kernel = PeriodicKernel()
    else:
        raise ValueError(f'Invalid kernel {args.bo_kernel}')

    for ind_seed in range(1, num_all + 1):
        seed_ = seed * ind_seed

        if seed_ is not None:
            torch.manual_seed(seed_)
            torch.cuda.manual_seed(seed_)

        if os.path.exists(get_str_file('./results', args.bo_kernel, 'oracle', args.t_noise, seed=ind_seed)):
            dict_exp = np.load(get_str_file('./results', args.bo_kernel, 'oracle', args.t_noise, seed=ind_seed), allow_pickle=True)
            dict_exp = dict_exp[()]
            list_dict.append(dict_exp)

            print(dict_exp)
            print(dict_exp['global'])
            print(np.array2string(dict_exp['minima'], separator=','))
            print(np.array2string(dict_exp['regrets'], separator=','))

            continue

        sampler = GPPriorSampler(kernel, t_noise=args.t_noise)

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
            print('ind_seed {} seed {} iter {}'.format(ind_seed, seed_, ind_iter + 1))

            cov_X_X, inv_cov_X_X, hyps = bayesogp.get_optimized_kernel(X_train, Y_train, None, str_cov, is_fixed_noise=False, debug=False)

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
            'seed': seed_,
            'str_cov': str_cov,
            'global': min_yp.cpu().numpy(),
            'minima': np.array(list_min),
            'regrets': np.array(list_min) - min_yp.cpu().numpy(),
            'xc': X_train,
            'yc': Y_train,
            'model': 'oracle',
        }

        np.save(get_str_file('./results', args.bo_kernel, 'oracle', args.t_noise, seed=ind_seed), dict_exp)
        list_dict.append(dict_exp)

    np.save(get_str_file('./figures/results', args.bo_kernel, 'oracle', args.t_noise), list_dict)

def bo(args, model):
    if args.mode == 'bo':
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)

    if args.bo_kernel == 'rbf':
        kernel = RBFKernel()
    elif args.bo_kernel == 'matern':
        kernel = Matern52Kernel()
    elif args.bo_kernel == 'periodic':
        kernel = PeriodicKernel()
    else:
        raise ValueError(f'Invalid kernel {args.bo_kernel}')

    seed = 42
    str_cov = 'se'
    num_all = 100
    num_iter = 50
    num_init = args.bo_num_init

    list_dict = []

    for ind_seed in range(1, num_all + 1):
        seed_ = seed * ind_seed

        if seed_ is not None:
            torch.manual_seed(seed_)
            torch.cuda.manual_seed(seed_)

        obj_prior = GPPriorSampler(kernel, t_noise=args.t_noise)

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
            print('ind_seed {} seed {} iter {}'.format(ind_seed, seed_, ind_iter + 1))

            with torch.no_grad():
                outs = model(batch, num_samples=args.bo_num_samples)
                print('ctx_ll {:.4f} tar ll {:.4f}'.format(
                    outs.ctx_ll.item(), outs.tar_ll.item()))
                py = model.predict(batch.xc, batch.yc,
                        xp[None,:,None].repeat(1, 1, 1),
                        num_samples=args.bo_num_samples)
                mu, sigma = py.mean.squeeze(0), py.scale.squeeze(0)

            if mu.dim() == 4:
                print(mu.shape, sigma.shape)
                var = sigma.pow(2).mean(0) + mu.pow(2).mean(0) - mu.mean(0).pow(2)
                sigma = var.sqrt().squeeze(0)
                mu = mu.mean(0).squeeze(0)
                mu_ = mu.cpu().numpy()
                sigma_ = sigma.cpu().numpy()

                acq_vals = -1.0 * acquisition.ei(np.ravel(mu_), np.ravel(sigma_), Y_train)

#                acq_vals = []

#                for ind_mu in range(0, mu.shape[0]):
#                    acq_vals_ = -1.0 * acquisition.ei(np.ravel(mu[ind_mu].cpu().numpy()), np.ravel(sigma[ind_mu].cpu().numpy()), Y_train)
#                    acq_vals.append(acq_vals_)

#                acq_vals = np.mean(acq_vals, axis=0)
            else:
                mu_ = mu.cpu().numpy()
                sigma_ = sigma.cpu().numpy()

                acq_vals = -1.0 * acquisition.ei(np.ravel(mu_), np.ravel(sigma_), Y_train)

#                var = sigma.pow(2).mean(0) + mu.pow(2).mean(0) - mu.mean(0).pow(2)
#                sigma = var.sqrt().squeeze(0)
#                mu = mu.mean(0).squeeze(0)

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
            'seed': seed_,
            'global': min_yp.cpu().numpy(),
            'minima': np.array(list_min),
            'regrets': np.array(list_min) - min_yp.cpu().numpy(),
            'xc': X_train,
            'yc': Y_train,
            'model': args.model,
            'cov': str_cov,
        }

        list_dict.append(dict_exp)

    np.save(get_str_file('./figures/results', args.bo_kernel, args.model, args.t_noise), list_dict)

if __name__ == '__main__':
    main()

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

from data.image import img_to_task, task_to_img
from data.emnist import EMNIST

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

    parser.add_argument('--max_num_points', type=int, default=200)
    parser.add_argument('--class_range', type=int, nargs='*', default=[0,10])

    parser.add_argument('--model', type=str, default='cnp')
    parser.add_argument('--train_batch_size', type=int, default=100)
    parser.add_argument('--train_num_samples', type=int, default=4)

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)

    parser.add_argument('--eval_seed', type=int, default=42)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_num_samples', type=int, default=50)
    parser.add_argument('--eval_logfile', type=str, default=None)

    parser.add_argument('--plot_seed', type=int, default=None)
    parser.add_argument('--plot_batch_size', type=int, default=16)
    parser.add_argument('--plot_num_samples', type=int, default=30)
    parser.add_argument('--plot_num_ctx', type=int, default=100)

    # OOD settings
    parser.add_argument('--t_noise', type=float, default=None)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_cls = getattr(load_module(f'models/{args.model}.py'), args.model.upper())
    with open(f'configs/emnist/{args.model}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model = model_cls(**config).cuda()

    args.root = osp.join(results_path, 'emnist', args.model, args.expid)

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

    train_ds = EMNIST(train=True, class_range=args.class_range)
    eval_ds = EMNIST(train=False, class_range=args.class_range)
    train_loader = torch.utils.data.DataLoader(train_ds,
        batch_size=args.train_batch_size,
        shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader)*args.num_epochs)

    if args.resume:
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_epoch = ckpt.epoch
    else:
        logfilename = osp.join(args.root, 'train_{}.log'.format(
            time.strftime('%Y%m%d-%H%M')))
        start_epoch = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    if not args.resume:
        logger.info('Total number of parameters: {}\n'.format(
            sum(p.numel() for p in model.parameters())))

    for epoch in range(start_epoch, args.num_epochs+1):
        model.train()
        for (x, _) in tqdm(train_loader):
            batch = img_to_task(x,
                    max_num_points=args.max_num_points,
                    device='cuda')
            optimizer.zero_grad()
            outs = model(batch, num_samples=args.train_num_samples)
            outs.loss.backward()
            optimizer.step()
            scheduler.step()

            for key, val in outs.items():
                ravg.update(key, val)

        line = f'{args.model}:{args.expid} epoch {epoch} '
        line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
        line += ravg.info()
        logger.info(line)

        if epoch % args.eval_freq == 0:
            logger.info(eval(args, model) + '\n')

        ravg.reset()

        if epoch % args.save_freq == 0 or epoch == args.num_epochs:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.epoch = epoch + 1
            torch.save(ckpt, osp.join(args.root, 'ckpt.tar'))

    args.mode = 'eval'
    eval(args, model)

def gen_evalset(args):

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    eval_ds = EMNIST(train=False, class_range=args.class_range)
    eval_loader = torch.utils.data.DataLoader(eval_ds,
            batch_size=args.eval_batch_size,
            shuffle=False, num_workers=4)

    batches = []
    for x, _ in tqdm(eval_loader):
        batches.append(img_to_task(x,
            t_noise=args.t_noise,
            max_num_points=args.max_num_points))

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    path = osp.join(evalsets_path, 'emnist')
    if not osp.isdir(path):
        os.makedirs(path)

    c1, c2 = args.class_range
    filename = f'{c1}-{c2}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.tar'

    torch.save(batches, osp.join(path, filename))

def eval(args, model):
    if args.mode == 'eval':
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        if args.eval_logfile is None:
            c1, c2 = args.class_range
            eval_logfile = f'eval_{c1}-{c2}'
            if args.t_noise is not None:
                eval_logfile += f'_{args.t_noise}'
            eval_logfile += '.log'
        else:
            eval_logfile = args.eval_logfile
        filename = osp.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    else:
        logger = None

    path = osp.join(evalsets_path, 'emnist')
    c1, c2 = args.class_range
    filename = f'{c1}-{c2}'
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

    c1, c2 = args.class_range
    line = f'{args.model}:{args.expid} {c1}-{c2} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    line += ravg.info()

    if logger is not None:
        logger.info(line)

    return line

def ensemble(args, model):
    num_runs = 5
    models = []
    for i in range(num_runs):
        model_ = deepcopy(model)
        ckpt = torch.load(osp.join(results_path, 'emnist', args.model, f'run{i+1}', 'ckpt.tar'))
        model_.load_state_dict(ckpt['model'])
        model_.cuda()
        model_.eval()
        models.append(model_)

    path = osp.join(evalsets_path, 'emnist')
    c1, c2 = args.class_range
    filename = f'{c1}-{c2}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.tar'
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)

    eval_batches = torch.load(osp.join(path, filename))

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

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    filename = f'ensemble_{c1}-{c2}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.log'
    logger = get_logger(osp.join(results_path, 'emnist', args.model, filename), mode='w')
    logger.info(ravg.info())

if __name__ == '__main__':
    main()

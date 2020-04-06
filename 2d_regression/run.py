import os
import argparse
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from attrdict import AttrDict

import torch
import torch.nn as nn
from torchvision.utils import make_grid

from data.image import img_to_task, task_to_img
from utils.misc import load_module
from utils.log import get_logger, RunningAverage

ROOT = '/nfs/parker/ext01/john/neural_process/2d_regression'

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',
            choices=['train', 'eval', 'plot', 'valid'],
            default='train')
    parser.add_argument('--expid', '-eid', type=str, default='trial')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='0')

    parser.add_argument('--max_num_points', '-mnp', type=int, default=200)

    parser.add_argument('--model', type=str, default='cnp')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--train_batch_size', '-tb', type=int, default=100)

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)

    parser.add_argument('--eval_log_file', '-elf', type=str, default=None)
    parser.add_argument('--eval_batch_size', '-eb', type=int, default=16)
    parser.add_argument('--eval_seed', type=int, default=42)
    parser.add_argument('--num_eval_samples', type=int, default=100)
    parser.add_argument('--heavy_tailed_noise', '-tn', type=float, default=0.0)

    parser.add_argument('--plot_seed', type=int, default=None)
    parser.add_argument('--plot_num_ctx', '-pnc', type=int, default=None)
    parser.add_argument('--plot_batch_size', type=int, default=16)
    parser.add_argument('--num_plot_samples', type=int, default=30)

    parser.add_argument('--valid_log_file', '-vlf', type=str, default=None)

    args, cmdline = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.root = os.path.join(ROOT, args.model, args.data, args.expid)

    # load data sampler
    data_file = 'data/{}.py'.format(args.data)
    (train_ds, eval_ds), cmdline = load_module(data_file).load(args, cmdline)

    # load model
    model_file = 'models/{}.py'.format(args.model)
    model, cmdline = load_module(model_file).load(args, cmdline)

    if len(cmdline) > 0:
        raise ValueError('unexpected arguments: {}'.format(cmdline))

    model.cuda()

    if args.mode == 'train':
        train_loader = torch.utils.data.DataLoader(train_ds,
                batch_size=args.train_batch_size, shuffle=True)
        eval_loader = torch.utils.data.DataLoader(eval_ds,
                batch_size=args.eval_batch_size, shuffle=False)
        train(args, train_loader, eval_loader, model)
    elif args.mode == 'eval':
        eval_loader = torch.utils.data.DataLoader(eval_ds,
                batch_size=args.eval_batch_size, shuffle=False)
        eval(args, eval_loader, model)
    elif args.mode == 'plot':
        plot_loader = torch.utils.data.DataLoader(eval_ds,
                batch_size=args.plot_batch_size, shuffle=True)
        plot(args, plot_loader, model)
    elif args.mode == 'valid':
        eval_loader = torch.utils.data.DataLoader(eval_ds,
                batch_size=args.eval_batch_size, shuffle=False)
        validate(args, eval_loader, model)

def train(args, train_loader, eval_loader, model):
    if not os.path.isdir(args.root):
        os.makedirs(args.root)

    with open(os.path.join(args.root, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, sort_keys=True, indent=4)
        print(json.dumps(args.__dict__, sort_keys=True, indent=4))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader)*args.num_epochs)

    if args.resume:
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_epoch = ckpt.epoch
    else:
        logfilename = os.path.join(args.root, 'train_{}.log'.format(
            time.strftime('%Y%m%d-%H%M')))
        start_epoch = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    if not args.resume:
        logger.info('Total number of parameters: {}\n'.format(
            sum(p.numel() for p in model.parameters())))

    for epoch in range(start_epoch, args.num_epochs+1):
        model.train()
        for x, _ in train_loader:
            batch = img_to_task(x, max_num_points=args.max_num_points)
            optimizer.zero_grad()
            outs = model(batch)
            outs.loss.backward()
            optimizer.step()
            scheduler.step()

            for key, val in outs.items():
                ravg.update(key, val)

        line = '{}:{}:{} epoch {} lr {:.3e} '.format(
                args.model, args.data, args.expid, epoch,
                optimizer.param_groups[0]['lr'])
        line += ravg.info()
        logger.info(line)

        if epoch % args.eval_freq == 0:
            logger.info(eval(args,  eval_loader, model) + '\n')

        ravg.reset()

        if epoch % args.save_freq == 0 or epoch == args.num_epochs:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.epoch = epoch + 1
            torch.save(ckpt, os.path.join(args.root, 'ckpt.tar'))

    args.mode = 'eval'
    eval(args, eval_loader, model)

def eval(args, eval_loader, model):
    if args.mode == 'eval':
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
    ravg = RunningAverage()

    # fix seed to get consistent eval sets
    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    model.eval()
    with torch.no_grad():
        for x, _ in eval_loader:
            batch = img_to_task(x,
                    max_num_points=args.max_num_points,
                    heavy_tailed_noise=args.heavy_tailed_noise)
            outs = model(batch, num_samples=args.num_eval_samples)
            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = '{}:{}:{} eval '.format(
            args.model, args.data, args.expid)
    line += ravg.info()

    if args.mode == 'eval':
        if args.eval_log_file is None:
            filename = '{}_'.format(args.data)
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

def plot(args, plot_loader, model):
    ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
    model.load_state_dict(ckpt.model)

    if args.plot_seed is not None:
        torch.manual_seed(args.plot_seed)
        torch.cuda.manual_seed(args.plot_seed)

    x, _ = next(iter(plot_loader))
    batch = img_to_task(x,
            num_ctx=args.plot_num_ctx,
            target_all=True,
            heavy_tailed_noise=args.heavy_tailed_noise)

    model.eval()
    with torch.no_grad():
        py = model.predict(batch.xc, batch.yc, batch.xt,
                num_samples=args.num_plot_samples)

    plt.figure('original')

    nrows = max(args.plot_batch_size//4, 1)
    img = make_grid(x, nrow=nrows)
    img = img.cpu().data.numpy().transpose(1, 2, 0)
    plt.imshow(img)

    yt_pred = py.sample()[0]
    #if yt_pred.dim() > 3:
    #    yt_pred = yt_pred.mean(0)
    plt.figure('model')
    mx = task_to_img(batch.xc, batch.yc, batch.xt, yt_pred,
            shape=x.shape[1:])
    img = make_grid(mx, nrow=nrows)
    img = img.cpu().data.numpy().transpose(1, 2, 0)
    plt.imshow(img)

    plt.show()

def validate(args, eval_loader, model):
    ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
    model.load_state_dict(ckpt.model)
    ravg = RunningAverage()

    # fix seed to get consistent eval sets
    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    model.eval()
    with torch.no_grad():
        for x, _ in tqdm(eval_loader):
            batch = img_to_task(x,
                    max_num_points=args.max_num_points,
                    heavy_tailed_noise=args.heavy_tailed_noise)
            log_diffs = model.validate(batch, num_samples=args.num_eval_samples)
            ravg.update('log_diffs', log_diffs)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = '{}:{}:{} eval '.format(
            args.model, args.data, args.expid)
    line += ravg.info()

    if args.valid_log_file is None:
        filename = '{}_'.format(args.data)
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

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
from data.emnist import EMNIST

from utils.paths import results_path
from utils.misc import load_module
from utils.log import get_logger, RunningAverage

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',
            choices=['train', 'eval', 'plot', 'valid'],
            default='train')
    parser.add_argument('--expid', '-eid', type=str, default='trial')
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

    parser.add_argument('--plot_seed', type=int, default=None)
    parser.add_argument('--plot_batch_size', type=int, default=16)
    parser.add_argument('--plot_num_samples', type=int, default=30)
    parser.add_argument('--plot_num_ctx', type=int, default=100)

    # OOD settings
    parser.add_argument('--ood', action='store_true', default=None)
    parser.add_argument('--t_noise', type=float, default=0.05)

    args, cmdline = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.root = os.path.join(results_path, 'emnist', args.model, args.expid)

    model, cmdline = load_module(f'models/emnist/{args.model}.py').load(args, cmdline)

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

    train_ds = EMNIST(train=True, class_range=args.class_range)
    eval_ds = EMNIST(train=False, class_range=args.class_range)
    train_loader = torch.utils.data.DataLoader(train_ds,
        batch_size=args.train_batch_size,
        shuffle=True, num_workers=4)
    eval_loader = torch.utils.data.DataLoader(eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False, num_workers=4)

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
            logger.info(eval(args, model, eval_loader=eval_loader) + '\n')

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
    eval(args, model, eval_loader=eval_loader)

def eval(args, model, eval_loader=None):
    if args.mode == 'eval' and args.model != 'ensemble':
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)

    if eval_loader is None:
        eval_ds = EMNIST(train=False, class_range=args.class_range)
        eval_loader = torch.utils.data.DataLoader(eval_ds,
                batch_size=args.eval_batch_size,
                shuffle=False, num_workers=4)

    def _eval(eval_loader, t_noise=None):
        ravg = RunningAverage()

        # fix seed to get consistent eval sets
        torch.manual_seed(args.eval_seed)
        torch.cuda.manual_seed(args.eval_seed)

        model.eval()
        with torch.no_grad():
            for x, _ in tqdm(eval_loader):
                batch = img_to_task(x,
                        t_noise=t_noise,
                        max_num_points=args.max_num_points,
                        device='cuda')
                outs = model(batch, num_samples=args.eval_num_samples)
                for key, val in outs.items():
                    ravg.update(key, val)

        torch.manual_seed(time.time())
        torch.cuda.manual_seed(time.time())
        return ravg.info()

    line = f'{args.model}:{args.expid} ' + _eval(eval_loader)
    if args.ood:
        line += f'\n{args.model}:{args.expid} tn {args.t_noise} ' \
                + _eval(eval_loader, t_noise=args.t_noise)
        eval_ds = EMNIST(train=False, class_range=[10, 47])
        eval_loader = torch.utils.data.DataLoader(eval_ds,
                batch_size=args.eval_batch_size,
                shuffle=False, num_workers=4)
        line += f'\n{args.model}:{args.expid} unseen ' \
                + _eval(eval_loader)

    if args.mode == 'eval':
        filename = os.path.join(args.root, 'eval.log')
        logger = get_logger(filename, mode='w')
        logger.info(line)

    return line

def plot(args, model):
    ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
    model.load_state_dict(ckpt.model)

    eval_ds = EMNIST(train=False, class_range=args.class_range)
    plot_loader = torch.utils.data.DataLoader(eval_ds,
            batch_size=args.plot_batch_size,
            shuffle=True, num_workers=4)

    if args.plot_seed is not None:
        torch.manual_seed(args.plot_seed)
        torch.cuda.manual_seed(args.plot_seed)

    def _plot(figtitle, t_noise=None):
        x, _ = next(iter(plot_loader))
        batch = img_to_task(x,
                num_ctx=args.plot_num_ctx,
                target_all=True,
                t_noise=t_noise,
                device='cuda')

        model.eval()
        with torch.no_grad():
            py = model.predict(batch.xc, batch.yc, batch.xt,
                    num_samples=args.plot_num_samples)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), num=figtitle)

        nrows = max(args.plot_batch_size//4, 1)
        img = make_grid(x, nrow=nrows)
        img = img.cpu().data.numpy().transpose(1, 2, 0)
        axes[0].imshow(img)
        axes[0].axis('off')
        axes[0].set_title('original')

        yt_pred = py.sample()
        if yt_pred.dim() > 3:
            yt_pred = yt_pred.mean(0)

        ctx, recon = task_to_img(batch.xc, batch.yc, batch.xt, yt_pred,
                shape=x.shape[1:])

        img = make_grid(ctx, nrow=nrows)
        img = img.cpu().data.numpy().transpose(1, 2, 0)
        axes[1].imshow(img)
        axes[1].axis('off')
        axes[1].set_title('context')

        img = make_grid(recon, nrow=nrows)
        img = img.cpu().data.numpy().transpose(1, 2, 0)
        axes[2].imshow(img)
        axes[2].axis('off')
        axes[2].set_title('reconstructed')

        plt.tight_layout()

    _plot('without noise')
    if args.ood:
        _plot(f'with tn {args.t_noise}', t_noise=args.t_noise)

    plt.show()

if __name__ == '__main__':
    main()

import os
import argparse
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from attrdict import AttrDict

import torch
import torch.nn as nn

from data.cifar_fs import get_loader

from utils.paths import results_path
from utils.misc import load_module, set_seed
from utils.log import get_logger, RunningAverage

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',
            choices=['train', 'eval'],
            default='train')
    parser.add_argument('--expid', type=str, default='trial')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='0')

    parser.add_argument('--ways', type=int, default=5)
    parser.add_argument('--shots', type=int, default=5)
    parser.add_argument('--test_shots', type=int, default=15)

    parser.add_argument('--model', type=str, default='proto')
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--train_num_samples', type=int, default=4)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_steps', type=int, default=20000)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=1000)

    parser.add_argument('--eval_seed', type=int, default=42)
    parser.add_argument('--eval_num_batches', type=int, default=1000)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_num_samples', type=int, default=50)
    parser.add_argument('--eval_logfile', type=str, default=None)

    args, cmdline = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model, cmdline = load_module(f'models/{args.model}.py').load(args, cmdline)
    if not hasattr(args, 'root'):
        args.root = os.path.join(results_path,
                'cifar_fs', f'{args.ways}_{args.shots}',
                args.model, args.expid)

    if len(cmdline) > 0:
        raise ValueError('unexpected arguments: {}'.format(cmdline))

    model.cuda()

    if args.mode == 'train':
        train(args, model)
    elif args.mode == 'eval':
        eval(args, model)

def train(args, model):
    if not os.path.isdir(args.root):
        os.makedirs(args.root)

    with open(os.path.join(args.root, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, sort_keys=True, indent=4)
        print(json.dumps(args.__dict__, sort_keys=True, indent=4))

    train_loader = get_loader(args.train_batch_size,
            args.ways, args.shots, args.test_shots,
            meta_split='train')
    eval_loader = get_loader(args.eval_batch_size,
            args.ways, args.shots, args.test_shots,
            meta_split='test')

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

    for step, batch in enumerate(train_loader, start_step):
        model.train()
        optimizer.zero_grad()
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
                line = eval(args, model, eval_loader=eval_loader)
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

        if step == args.num_steps:
            break

    args.mode = 'eval'
    eval(args, model)

def eval(args, model, eval_loader=None):

    if args.mode == 'eval' and args.model != 'ensemble':
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)

    if eval_loader is None:
        eval_loader = get_loader(args.eval_batch_size,
                args.ways, args.shots, args.test_shots,
                meta_split='test')

    def _eval(eval_loader):
        ravg = RunningAverage()
        set_seed(args.eval_seed)
        model.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(eval_loader, 1)):
                outs = model(batch)
                for key, val in outs.items():
                    ravg.update(key, val)
        set_seed()
        return ravg.info()

    line = f'{args.model}:{args.expid} ' + _eval(eval_loader)

    if args.mode == 'eval':
        filename = os.path.join(args.root, 'eval.log')
        logger = get_logger(filename, mode='w')
        logger.info(line)

    return line

if __name__ == '__main__':
    main()

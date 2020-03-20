import argparse
import json
import os
import pickle
import time
from importlib.machinery import SourceFileLoader
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from attrdict import AttrDict
from torch.utils.tensorboard import SummaryWriter

from log import RunningAverage, get_logger


def train(args, sampler, model):
    if not os.path.isdir(args.root):
        os.makedirs(args.root)

    with open(os.path.join(args.root, "args.json"), "w") as f:
        json.dump(args.__dict__, f, sort_keys=True, indent=4)
        # print(json.dumps(args.__dict__, sort_keys=True, indent=4))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps
    )

    if args.resume:
        ckpt = torch.load(os.path.join(args.root, "ckpt.tar"))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_step = ckpt.step
    else:
        logfilename = os.path.join(
            args.root, "train_{}.log".format(time.strftime("%Y%m%d-%H%M"))
        )
        start_step = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    start_time = timer()
    for step in range(start_step, args.num_steps + 1):
        if timer() - start_time > args.time_budget:
            print(f"Stopping at {step=}")
            ckpt = AttrDict() 
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.step = step + 1
            torch.save(ckpt, os.path.join(args.root, "ckpt.tar"))
            break

        model.train()
        optimizer.zero_grad()
        batch = sampler.sample(
            batch_size=args.train_batch_size,
            max_num_points=args.max_num_points,
            heavy_tailed_noise=args.heavy_tailed_noise,
            device="cuda",
        )
        outs = model(batch)
        outs.loss.backward()
        optimizer.step()
        scheduler.step()

        for key, val in outs.items():
            ravg.update(key, val)

        if step % args.print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            line = "{}:{}:{} step {} lr {:.3e} ".format(
                args.model, args.train_data, args.expid, step, lr
            )
            writer.add_scalar("lr", lr, global_step=step)
            writer.add_scalar("train_ll", ravg.get("ll"), global_step=step)
            writer.add_scalar("train_loss", ravg.get("loss"), global_step=step)
            line += ravg.info()
            logger.info(line)

            if step % args.eval_freq == 0:
                logger.info(eval(args, sampler, model) + "\n")

            ravg.reset()

        if step % args.save_freq == 0 or step == args.num_steps:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.step = step + 1
            torch.save(ckpt, os.path.join(args.root, "ckpt.tar"))

    args.mode = "eval"
    eval(args, sampler, model)


def eval(args, sampler, model):
    args.epoch += 1
    if args.mode == "eval":
        ckpt = torch.load(os.path.join(args.root, "ckpt.tar"))
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
                device="cuda",
            )
            outs = model(batch, num_samples=args.num_eval_samples)
            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = "{}:{}:{} eval ".format(args.model, args.eval_data, args.expid)
    line += ravg.info()

    pred_ll = ravg.get("pred_ll")
    writer.add_scalar("pred_ll", pred_ll, global_step=args.epoch)
    with open(f"{args.root}/metrics.pkl", "wb") as handle:
        output_dict = dict(args.__dict__, **{"pred_ll": pred_ll})
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if args.mode == "eval":
        if args.eval_log is None:
            filename = "{}_".format(args.eval_data)
            if args.heavy_tailed_noise:
                filename += "htn_"
            filename += "eval.log"
        else:
            filename = args.eval_log

        filename = os.path.join(args.root, filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logger = get_logger(filename, mode="w")
        logger.info(line)

    return line


def plot(args, sampler, model):
    def tnp(x):
        return x.squeeze().cpu().data.numpy()

    if args.mode == "plot":
        ckpt = torch.load(os.path.join(args.root, "ckpt.tar"))
        model.load_state_dict(ckpt.model)

    if args.plot_seed is not None:
        torch.manual_seed(args.plot_seed)
        torch.cuda.manual_seed(args.plot_seed)

    batch = sampler.sample(
        batch_size=args.plot_batch_size,
        max_num_points=args.max_num_points,
        heavy_tailed_noise=args.heavy_tailed_noise,
        device="cuda",
    )

    xp = torch.linspace(-2, 2, 200).cuda()
    model.eval()

    with torch.no_grad():
        outs = model(batch, args.num_plot_samples)
        print(outs.pred_ll.item())
        py = model.predict(
            batch.xc,
            batch.yc,
            xp[None, :, None].repeat(args.plot_batch_size, 1, 1),
            num_samples=args.num_plot_samples,
        )
        mu, sigma = py.mean.squeeze(0), py.scale.squeeze(0)

    if args.plot_batch_size > 1:
        nrows = max(args.plot_batch_size // 4, 1)
        ncols = min(4, args.plot_batch_size)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        axes = axes.flatten()
    else:
        fig = plt.figure(figsize=(5, 5))
        axes = [plt.gca()]

    # multi sample
    if mu.dim() == 4:
        for i, ax in enumerate(axes):
            for s in range(mu.shape[0]):
                ax.plot(
                    tnp(xp),
                    tnp(mu[s][i]),
                    color="steelblue",
                    alpha=max(0.5 / args.num_plot_samples, 0.1),
                )
                ax.fill_between(
                    tnp(xp),
                    tnp(mu[s][i]) - tnp(sigma[s][i]),
                    tnp(mu[s][i]) + tnp(sigma[s][i]),
                    color="skyblue",
                    alpha=max(0.2 / args.num_plot_samples, 0.02),
                    linewidth=0.0,
                )
            ax.scatter(
                tnp(batch.xc[i]),
                tnp(batch.yc[i]),
                color="k",
                label="context",
                zorder=mu.shape[0] + 1,
            )
            ax.scatter(
                tnp(batch.xt[i]),
                tnp(batch.yt[i]),
                color="orchid",
                label="target",
                zorder=mu.shape[0] + 1,
            )
            ax.legend()
    else:
        for i, ax in enumerate(axes):
            ax.plot(tnp(xp), tnp(mu[i]), color="steelblue", alpha=0.5)
            ax.fill_between(
                tnp(xp),
                tnp(mu[i] - sigma[i]),
                tnp(mu[i] + sigma[i]),
                color="skyblue",
                alpha=0.2,
                linewidth=0.0,
            )
            ax.scatter(tnp(batch.xc[i]), tnp(batch.yc[i]), color="k", label="context")
            ax.scatter(
                tnp(batch.xt[i]), tnp(batch.yt[i]), color="orchid", label="target"
            )
            ax.legend()

    plt.tight_layout()
    plt.show()


parser = argparse.ArgumentParser()

parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--expid", type=str, default="trial")
parser.add_argument("--resume", action="store_true", default=False)

parser.add_argument("--mode", choices=["train", "eval", "plot"], default="train")

parser.add_argument(
    "--model",
    type=str,
    choices=["anp", "banp", "bnp", "canp", "cnp", "np"],
    default="cnp",
)
# for bootstrap models
parser.add_argument("--r_bs", type=float, default=0.0)

parser.add_argument("--fixed_var", "-fv", action="store_true", default=False)
parser.add_argument("--heavy_tailed_noise", "-htn", action="store_true", default=False)

parser.add_argument("--max_num_points", "-mnp", type=int, default=50)

parser.add_argument("--train_data", "-td", type=str, default="rbf")
parser.add_argument("--train_batch_size", "-tb", type=int, default=100)

parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--num_steps", type=int, default=100000)
parser.add_argument("--print_freq", type=int, default=200)
parser.add_argument("--eval_freq", type=int, default=5000)
parser.add_argument("--save_freq", type=int, default=1000)

parser.add_argument("--eval_data", "-ed", type=str, default="rbf")
parser.add_argument("--eval_log", "-el", type=str, default=None)
parser.add_argument("--eval_batch_size", "-eb", type=int, default=16)
parser.add_argument("--eval_seed", type=int, default=42)
parser.add_argument("--num_eval_batches", type=int, default=1000)
parser.add_argument("--num_eval_samples", type=int, default=100)

parser.add_argument("--plot_seed", type=int, default=None)
parser.add_argument("--plot_batch_size", type=int, default=16)
parser.add_argument("--num_plot_samples", type=int, default=30)
parser.add_argument("--time_budget", type=int, default=60 * 60)  # 1 hr budget

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.root = f"result/{args.model}_{args.train_data}_{args.expid}"
args.root += f"lr{args.lr}_bs{args.train_batch_size}"
writer = SummaryWriter(log_dir=args.root)
args.epoch = 0

# load data sampler
datamodule = args.train_data if args.mode == "train" else args.eval_data
sampler = (
    SourceFileLoader(datamodule, os.path.join("data/{}.py".format(datamodule)))
    .load_module()
    .load(args)
)

# load model
model = (
    SourceFileLoader(args.model, os.path.join("models/{}.py".format(args.model)))
    .load_module()
    .load(args)
)
model.cuda()

if args.mode == "train":
    train(args, sampler, model)
elif args.mode == "eval":
    eval(args, sampler, model)
elif args.mode == "plot":
    plot(args, sampler, model)

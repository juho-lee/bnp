import torch
import matplotlib.pyplot as plt
import os.path as osp

from data.gp import *

from models.gp.anp import ANP
from models.gp.rbnp import RBNP

from utils.paths import results_path
from utils.misc import load_module

expid = 'run1'
num_ctx = 15
max_num_points = 50
seed = 23
p = 0.5
tn = 0.5
K = 50

# load anp
anp_model = ANP().cuda()
anp_model.eval()
ckpt = torch.load(osp.join(results_path, 'gp', 'anp', expid, 'ckpt.tar'))
anp_model.load_state_dict(ckpt.model)

# load rbnp-anp
rbnp_model = RBNP(ANP()).cuda()
rbnp_model.eval()
ckpt = torch.load(osp.join(results_path, 'gp', 'rbnp', 'anp', expid, 'ckpt.tar'))
rbnp_model.load_state_dict(ckpt.model)

#sampler = GPSampler(PeriodicKernel(p=p))
sampler = GPSampler(RBFKernel(), t_noise=tn)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
batch = sampler.sample(batch_size=1,
        num_ctx=num_ctx,
        max_num_points=max_num_points,
        device='cuda')

with torch.no_grad():
    outs = anp_model(batch, num_samples=K)
    print(f'anp {outs.ctx_ll.item():.3f} {outs.tar_ll.item():.3f}')
    outs = rbnp_model(batch, num_samples=K)
    print(f'rbnp {outs.ctx_ll.item():.3f} {outs.tar_ll.item():.3f}')

with torch.no_grad():
    xp = torch.linspace(-2, 2, 200).cuda()
    py = anp_model.predict(batch.xc, batch.yc,
            xp[None,:,None].repeat(1, 1, 1),
            num_samples=K)
    mu, sigma = py.mean, py.scale
    var = sigma.pow(2).mean(0) + mu.pow(2).mean(0) - mu.mean(0).pow(2)
    anp_sigma = var.sqrt()
    anp_mu = mu.mean(0)

    py = rbnp_model.predict(batch.xc, batch.yc,
            xp[None,:,None].repeat(1, 1, 1),
            num_samples=K)
    mu, sigma = py.mean, py.scale
    var = sigma.pow(2).mean(0) + mu.pow(2).mean(0) - mu.mean(0).pow(2)
    rbnp_sigma = var.sqrt()
    rbnp_mu = mu.mean(0)

def tnp(x):
    return x.squeeze().cpu().data.numpy()

plt.figure(figsize=(7, 7))
plt.scatter(tnp(batch.xc[0]), tnp(batch.yc[0]), color='k', zorder=3)
plt.scatter(tnp(batch.xt[0]), tnp(batch.yt[0]), color='orchid', zorder=3)
plt.plot(tnp(xp), tnp(anp_mu[0]), color='steelblue', alpha=0.5, zorder=2)
plt.fill_between(tnp(xp), tnp(anp_mu[0]-anp_sigma[0]), tnp(anp_mu[0]+anp_sigma[0]),
        color='skyblue', alpha=0.3, linewidth=0.0, zorder=2)

plt.plot(tnp(xp), tnp(rbnp_mu[0]), color='green', alpha=0.5, zorder=1)
plt.fill_between(tnp(xp), tnp(rbnp_mu[0]-rbnp_sigma[0]), tnp(rbnp_mu[0]+rbnp_sigma[0]),
        color='green', alpha=0.1, linewidth=0.0, zorder=1)
plt.show()

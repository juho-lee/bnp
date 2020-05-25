import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, StudentT
from attrdict import AttrDict
import math

__all__ = ['GPSampler', 'RBFKernel', 'PeriodicKernel', 'Matern52Kernel']

class GPSampler(object):
    def __init__(self, kernel, t_noise=None):
        self.kernel = kernel
        self.t_noise = t_noise

    def sample(self,
            batch_size=16,
            num_ctx=None,
            max_num_points=50,
            x_range=(-2, 2),
            device='cpu'):

        batch = AttrDict()
        num_ctx = num_ctx or torch.randint(low=3, high=max_num_points-3, size=[1]).item()
        num_tar = torch.randint(low=3, high=max_num_points-num_ctx, size=[1]).item()

        num_points = num_ctx + num_tar
        batch.x = x_range[0] + (x_range[1] - x_range[0]) \
                * torch.rand([batch_size, num_points, 1], device=device)
        batch.xc = batch.x[:,:num_ctx]
        batch.xt = batch.x[:,num_ctx:]

        # batch_size * num_points * num_points
        cov = self.kernel(batch.x)
        mean = torch.zeros(batch_size, num_points, device=device)
        batch.y = MultivariateNormal(mean, cov).rsample().unsqueeze(-1)
        batch.yc = batch.y[:,:num_ctx]
        batch.yt = batch.y[:,num_ctx:]

        if self.t_noise is not None:
            if self.t_noise == -1:
                t_noise = 0.15 * torch.rand(batch.y.shape).to(device)
            else:
                t_noise = self.t_noise
            batch.y += t_noise * StudentT(2.1).rsample(batch.y.shape).to(device)
        return batch

class RBFKernel(object):
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: batch_size * num_points * dim
    def __call__(self, x):
        length = 0.1 + (self.max_length-0.1) \
                * torch.rand([x.shape[0], 1, 1, 1], device=x.device)
        scale = 0.1 + (self.max_scale-0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)

        # batch_size * num_points * num_points * dim
        dist = (x.unsqueeze(-2) - x.unsqueeze(-3))/length

        # batch_size * num_points * num_points
        cov = scale.pow(2) * torch.exp(-0.5 * dist.pow(2).sum(-1)) \
                + self.sigma_eps**2 * torch.eye(x.shape[-2]).to(x.device)

        return cov

class Matern52Kernel(object):
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: batch_size * num_points * dim
    def __call__(self, x):
        length = 0.1 + (self.max_length-0.1) \
                * torch.rand([x.shape[0], 1, 1, 1], device=x.device)
        scale = 0.1 + (self.max_scale-0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)

        # batch_size * num_points * num_points
        dist = torch.norm((x.unsqueeze(-2) - x.unsqueeze(-3))/length, dim=-1)

        cov = scale.pow(2)*(1 + math.sqrt(5.0)*dist + 5.0*dist.pow(2)/3.0) \
                * torch.exp(-math.sqrt(5.0) * dist) \
                + self.sigma_eps**2 * torch.eye(x.shape[-2]).to(x.device)

        return cov

class PeriodicKernel(object):
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        #self.p = p
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: batch_size * num_points * dim
    def __call__(self, x):
        p = 0.1 + 0.4*torch.rand([x.shape[0], 1, 1], device=x.device)
        length = 0.1 + (self.max_length-0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)
        scale = 0.1 + (self.max_scale-0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)

        dist = x.unsqueeze(-2) - x.unsqueeze(-3)
        cov = scale.pow(2) * torch.exp(\
                - 2*(torch.sin(math.pi*dist.abs().sum(-1)/p)/length).pow(2)) \
                + self.sigma_eps**2 * torch.eye(x.shape[-2]).to(x.device)

        return cov

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, StudentT
from attrdict import AttrDict
import math

class GPSampler(object):
    def __init__(self, kernel):
        self.kernel = kernel

    def sample(self,
            batch_size=16,
            max_num_points=50,
            x_range=(-2, 2),
            heavy_tailed_noise=False,
            device='cpu'):

        batch = AttrDict()
        num_ctx = torch.randint(low=3, high=max_num_points-3, size=[1]).item()
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

        if heavy_tailed_noise:
            batch.yc += 0.1*StudentT(2.0).rsample(batch.yc.shape).to(device)

        return batch

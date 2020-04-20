import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, StudentT
from attrdict import AttrDict
import math

class GPPriorSampler(object):
    def __init__(self, kernel):
        self.kernel = kernel

    def sample(self,
            bx,
            heavy_tailed_noise=0.0,
            device='cpu'):
        # bx: 1 * num_points * 1

        # 1 * num_points * num_points
        cov = self.kernel(bx)
        mean = torch.zeros(1, bx.shape[1], device=device)
        mean = mean.cuda()

        by = MultivariateNormal(mean, cov).rsample().unsqueeze(-1)

        if heavy_tailed_noise > 0:
            by += heavy_tailed_noise * \
                    StudentT(2.0).rsample(by.shape).to(device)

        return by

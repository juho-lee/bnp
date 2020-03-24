import torch
from torch.distributions import Dirichlet
import math

def sample_bootstrap(*args, num_samples=None, r_bs=1.0, r_N=1.0):
    B, N, _ = args[0].shape
    K = num_samples or 1
    N = max(1, math.ceil(N * r_N))

    N_bs = math.ceil(N * r_bs)
    N_fixed = N - N_bs
    if N_bs == 0:
        bs_args = []
        for arg in args:
            bs_args.append(torch.stack([arg]*K))
        return bs_args
    else:
        if N_fixed == 0:
            idxs = torch.randint(N, size=[K, B, N, 1]).to(args[0].device)
        else:
            idxs = torch.rand(K, B, N, 1).argsort(-2)
            idxs1 = idxs[:,:,:N_fixed]
            idxs2 = torch.gather(idxs[:,:,N_fixed:], -2,
                    torch.randint(N_bs, size=[K, B, N_bs, 1]))
            idxs = torch.cat([idxs1, idxs2], -2).to(args[0].device)

        bs_args = []
        for arg in args:
            bs_args.append(
                    torch.gather(torch.stack([arg]*K), -2,
                        idxs.repeat(1, 1, 1, arg.shape[-1])).squeeze(0))
        return bs_args

def random_split(*args, num_samples=None, r=1.0):
    B, N, _ = args[0].shape
    K = num_samples or 1
    N_first = math.ceil(N * r)
    N_second = N - N_first
    idxs = torch.rand(K, B, N, 1).argsort(-2).to(args[0].device)
    idxs_first = idxs[:,:,:N_first]
    idxs_second = idxs[:,:,N_first:]
    first, second = [], []
    for arg in args:
        first.append(torch.gather(torch.stack([arg]*K), -2,
            idxs_first.repeat(1, 1, 1, arg.shape[-1])).squeeze(0))
        second.append(torch.gather(torch.stack([arg]*K), -2,
            idxs_second.repeat(1, 1, 1, arg.shape[-1])).squeeze(0))
    return first, second

import torch
import math

# items: list of batch_size * num_elements * dim tensors
# idxs: num_samples * batch_size * num_elements
def gather(items, idxs):
    K = idxs.shape[0]
    idxs = idxs.unsqueeze(-1).to(items[0].device)
    gathered_items = []
    for item in items:
        gathered_items.append(
                torch.gather(torch.stack([item]*K), -2,
                    idxs.repeat(1, 1, 1, item.shape[-1])).squeeze(0))
    return gathered_items

def sample_subset(items, r, num_samples=None):
    K = num_samples or 1
    B, N, _ = items[0].shape
    Ns = max(1, int(r * N))
    idxs = torch.rand(K, B, N).argsort(-1)
    return gather(items, idxs[...,:Ns]), gather(items, idxs[...,Ns:])

def sample_with_replacement(items, num_samples=None, r=1.0):
    K = num_samples or 1
    B, N, _ = items[0].shape
    Ns = max(1, int(r * N))
    idxs = torch.randint(N, size=[K, B, Ns])
    return gather(items, idxs)

def sample_with_partial_replacement(items, num_samples=None, r=1.0):
    if r == 1.0:
        return sample_with_replacement(items,
                num_samples=num_samples)
    elif r == 0.0:
        return items
    else:
        K = num_samples or 1
        B, N, _ = items[0].shape
        N_rep = max(1, int(r * N))
        idxs = torch.rand(K, B, N).argsort(-1)
        idxs_with_rep = torch.gather(idxs[..., :N_rep], -1,
                torch.randint(N_rep, size=[K, B, N_rep]))
        idxs_without_rep = idxs[..., N_rep:]
        idxs = torch.cat([idxs_with_rep, idxs_without_rep], -1)
        return gather(items, idxs)

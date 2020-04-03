import torch

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

def sample_subset(items, r_N=None, num_samples=None):
    r_N = r_N or torch.rand(1).item()
    K = num_samples or 1
    B, N = items[0].shape[-3], items[0].shape[-2]
    Ns = max(1, int(r_N * N))
    idxs = torch.rand(K, B, N).argsort(-1)
    return gather(items, idxs[...,:Ns]), gather(items, idxs[...,Ns:])

def sample_with_replacement(items, num_samples=None, r_N=1.0):
    K = num_samples or 1
    B, N = items[0].shape[-3], items[0].shape[-2]
    Ns = max(1, int(r_N * N))
    idxs = torch.randint(N, size=[K, B, Ns])
    return gather(items, idxs)

import torch

# args: tensors having shape B * N * dim
def sample_bootstrap(*args, num_samples=None):
    B, N, _ = args[0].shape
    K = num_samples or 1
    idxs = torch.randint(N, size=[K, B, N, 1]).to(args[0].device)
    bs_args = []
    for arg in args:
        bs_args.append(
                torch.gather(torch.stack([arg]*K), -2,
                    idxs.repeat(1, 1, 1, arg.shape[-1])).squeeze(0))
    return bs_args

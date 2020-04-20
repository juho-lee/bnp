import torch
from attrdict import AttrDict
from torch.utils.data import DataLoader
from torch.distributions import StudentT

def img_to_task(x,
        num_ctx=None,
        max_num_points=None,
        target_all=False,
        heavy_tailed_noise=0.0):

    B, C, H, W = x.shape
    x = x.view(B, C, -1)
    device = x.device
    batch = AttrDict()

    max_num_points = max_num_points or H*W
    num_ctx = torch.randint(low=3, high=max_num_points-3, size=[1]).item() \
            if num_ctx is None else num_ctx
    num_tar = max_num_points-num_ctx if target_all else \
            torch.randint(low=3, high=max_num_points-num_ctx, size=[1]).item()

    num_points = num_ctx + num_tar
    idxs = torch.rand(B, H*W).argsort(-1)[...,:num_points].to(device)
    x1, x2 = idxs//W, idxs%W
    batch.x = torch.stack([
        2*x1.float()/(H-1) - 1,
        2*x2.float()/(W-1) - 1], -1)
    batch.y = torch.gather(x, -1, idxs.unsqueeze(-2).repeat(1, C, 1))\
            .transpose(-2, -1) - 0.5

    batch.xc = batch.x[:,:num_ctx]
    batch.xt = batch.x[:,num_ctx:]
    batch.yc = batch.y[:,:num_ctx]
    batch.yt = batch.y[:,num_ctx:]

    if heavy_tailed_noise > 0:
        batch.y += heavy_tailed_noise * \
                StudentT(2.0).rsample(batch.y.shape).to(device)
        batch.y = batch.y.clamp(-0.5, 0.5)

    return batch

def task_to_img(xc, yc, xt, yt, shape):

    xc = xc.cpu()
    yc = yc.cpu()
    xt = xt.cpu()
    yt = yt.cpu()

    B = xc.shape[0]
    C, H, W = shape

    img = torch.zeros(B, C, H, W).to(xc.device)
    xc1, xc2 = xc[...,0], xc[...,1]
    xc1 = ((xc1+1)*(H-1)/2).round().long()
    xc2 = ((xc2+1)*(W-1)/2).round().long()

    xt1, xt2 = xt[...,0], xt[...,1]
    xt1 = ((xt1+1)*(H-1)/2).round().long()
    xt2 = ((xt2+1)*(W-1)/2).round().long()

    for b in range(B):
        for c in range(C):
            img[b,c,xc1[b],xc2[b]] = yc[b,:,c] + 0.5
            img[b,c,xt1[b],xt2[b]] = yt[b,:,c] + 0.5

    return img.clamp(0, 1)

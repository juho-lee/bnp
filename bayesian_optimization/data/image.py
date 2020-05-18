import torch
from attrdict import AttrDict
from torch.utils.data import DataLoader
from torch.distributions import StudentT

def img_to_task(img, num_ctx=None, max_num_points=None,
        target_all=False, t_noise=0.0, device=None):

    B, C, H, W = img.shape
    num_pixels = H*W
    img = img.view(B, C, -1)

    if t_noise is not None:
        img += t_noise * \
                StudentT(2.2).rsample(img.shape).to(img.device)

    device = img.device if device is None else device

    batch = AttrDict()
    max_num_points = max_num_points or num_pixels
    num_ctx = num_ctx or \
            torch.randint(low=3, high=max_num_points-3, size=[1]).item()
    num_tar = max_num_points - num_ctx if target_all else \
            torch.randint(low=3, high=max_num_points-num_ctx, size=[1]).item()
    num_points = num_ctx + num_tar
    idxs = torch.rand(B, num_pixels).argsort(-1)[...,:num_points].to(img.device)
    x1, x2 = idxs//W, idxs%W
    batch.x = torch.stack([
        2*x1.float()/(H-1) - 1,
        2*x2.float()/(W-1) - 1], -1).to(device)
    batch.y = (torch.gather(img, -1, idxs.unsqueeze(-2).repeat(1, C, 1))\
            .transpose(-2, -1) - 0.5).to(device)

    batch.xc = batch.x[:,:num_ctx]
    batch.xt = batch.x[:,num_ctx:]
    batch.yc = batch.y[:,:num_ctx]
    batch.yt = batch.y[:,num_ctx:]

    return batch

def task_to_img(xc, yc, xt, yt, shape):
    xc = xc.cpu()
    yc = yc.cpu()
    xt = xt.cpu()
    yt = yt.cpu()

    B = xc.shape[0]
    C, H, W = shape

    xc1, xc2 = xc[...,0], xc[...,1]
    xc1 = ((xc1+1)*(H-1)/2).round().long()
    xc2 = ((xc2+1)*(W-1)/2).round().long()

    xt1, xt2 = xt[...,0], xt[...,1]
    xt1 = ((xt1+1)*(H-1)/2).round().long()
    xt2 = ((xt2+1)*(W-1)/2).round().long()

    task_img = torch.zeros(B, 3, H, W).to(xc.device)
    task_img[:,2,:,:] = 1.0
    task_img[:,1,:,:] = 0.4
    for b in range(B):
        for c in range(3):
            task_img[b,c,xc1[b],xc2[b]] = yc[b,:,min(c,C-1)] + 0.5
    task_img = task_img.clamp(0, 1)

    completed_img = task_img.clone()
    for b in range(B):
        for c in range(3):
            completed_img[b,c,xt1[b],xt2[b]] = yt[b,:,min(c,C-1)] + 0.5
    completed_img = completed_img.clamp(0, 1)

    return task_img, completed_img

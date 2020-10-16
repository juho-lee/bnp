import torch
import numpy as np
import numpy.random as npr
import numba as nb
from tqdm import tqdm
from attrdict import AttrDict
#import pandas as pd
import wget

import os.path as osp
from utils.paths import datasets_path

@nb.njit(nb.i4(nb.f8[:]))
def catrnd(prob):
    cprob = prob.cumsum()
    u = npr.rand()
    for i in range(len(cprob)):
        if u < cprob[i]:
            return i
    return i

@nb.njit(nb.types.Tuple((nb.f8[:,:,:], nb.f8[:,:,:], nb.i4)) \
        (nb.i4, nb.i4, nb.i4, \
        nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.f8))
def _simulate_task(batch_size, num_steps, max_num_points,
        X0, Y0, theta0, theta1, theta2, theta3):

    time = np.zeros((batch_size, num_steps, 1))
    pop = np.zeros((batch_size, num_steps, 2))
    length = num_steps*np.ones((batch_size))

    for b in range(batch_size):
        pop[b,0,0] = max(int(X0 + npr.randn()), 1)
        pop[b,0,1] = max(int(Y0 + npr.randn()), 1)
        for i in range(1, num_steps):
            X, Y = pop[b,i-1,0], pop[b,i-1,1]
            rates = np.array([
                theta0*X*Y,
                theta1*X,
                theta2*Y,
                theta3*X*Y])
            total_rate = rates.sum()

            time[b,i,0] = time[b,i-1,0] + npr.exponential(scale=1./total_rate)

            pop[b,i,0] = pop[b,i-1,0]
            pop[b,i,1] = pop[b,i-1,1]
            a = catrnd(rates/total_rate)
            if a == 0:
                pop[b,i,0] += 1
            elif a == 1:
                pop[b,i,0] -= 1
            elif a == 2:
                pop[b,i,1] += 1
            else:
                pop[b,i,1] -= 1

            if pop[b,i,0] == 0 or pop[b,i,1] == 0:
                length[b] = i+1
                break

    num_ctx = npr.randint(15, max_num_points-15)
    num_tar = npr.randint(15, max_num_points-num_ctx)
    num_points = num_ctx + num_tar
    min_length = length.min()
    while num_points > min_length:
        num_ctx = npr.randint(15, max_num_points-15)
        num_tar = npr.randint(15, max_num_points-num_ctx)
        num_points = num_ctx + num_tar

    x = np.zeros((batch_size, num_points, 1))
    y = np.zeros((batch_size, num_points, 2))
    for b in range(batch_size):
        idxs = np.arange(int(length[b]))
        npr.shuffle(idxs)
        for j in range(num_points):
            x[b,j,0] = time[b,idxs[j],0]
            y[b,j,0] = pop[b,idxs[j],0]
            y[b,j,1] = pop[b,idxs[j],1]

    return x, y, num_ctx

class LotkaVolterraSimulator(object):
    def __init__(self,
            X0=50,
            Y0=100,
            theta0=0.01,
            theta1=0.5,
            theta2=1.0,
            theta3=0.01):

        self.X0 = X0
        self.Y0 = Y0
        self.theta0 = theta0
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3

    def simulate_tasks(self,
            num_batches,
            batch_size,
            num_steps=20000,
            max_num_points=100):

        batches = []
        for _ in tqdm(range(num_batches)):
            batch = AttrDict()
            x, y, num_ctx = _simulate_task(
                    batch_size, num_steps, max_num_points,
                    self.X0, self.Y0, self.theta0, self.theta1, self.theta2, self.theta3)
            batch.x = torch.Tensor(x)
            batch.y = torch.Tensor(y)
            batch.xc = batch.x[:,:num_ctx]
            batch.xt = batch.x[:,num_ctx:]
            batch.yc = batch.y[:,:num_ctx]
            batch.yt = batch.y[:,num_ctx:]

            batches.append(batch)

        return batches

def load_hare_lynx(num_batches, batch_size):

    filename = osp.join(datasets_path, 'lotka_volterra', 'LynxHare.txt')
    if not osp.isfile(filename):
        wget.download('http://people.whitman.edu/~hundledr/courses/M250F03/LynxHare.txt',
                out=osp.join(datsets_path, 'lotka_volterra'))

    tb = np.loadtxt(filename)
    times = torch.Tensor(tb[:,0]).unsqueeze(-1)
    pops = torch.stack([torch.Tensor(tb[:,2]), torch.Tensor(tb[:,1])], -1)

    #tb = pd.read_csv(osp.join(datasets_path, 'lotka_volterra', 'hare-lynx.csv'))
    #times = torch.Tensor(np.array(tb['time'])).unsqueeze(-1)
    #pops = torch.stack([torch.Tensor(np.array(tb['lynx'])),
    #    torch.Tensor(np.array(tb['hare']))], -1)

    batches = []
    N = pops.shape[-2]
    for _ in range(num_batches):
        batch = AttrDict()

        num_ctx = torch.randint(low=15, high=N-15, size=[1]).item()
        num_tar = N - num_ctx

        idxs = torch.rand(batch_size, N).argsort(-1)

        batch.x = torch.gather(
                torch.stack([times]*batch_size),
                -2, idxs.unsqueeze(-1))
        batch.y = torch.gather(torch.stack([pops]*batch_size),
                -2, torch.stack([idxs]*2, -1))
        batch.xc = batch.x[:,:num_ctx]
        batch.xt = batch.x[:,num_ctx:]
        batch.yc = batch.y[:,:num_ctx]
        batch.yt = batch.y[:,num_ctx:]

        batches.append(batch)

    return batches

if __name__ == '__main__':
    import argparse
    import os
    from utils.paths import datasets_path
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_batches', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--filename', type=str, default='batch')
    parser.add_argument('--X0', type=float, default=50)
    parser.add_argument('--Y0', type=float, default=100)
    parser.add_argument('--theta0', type=float, default=0.01)
    parser.add_argument('--theta1', type=float, default=0.5)
    parser.add_argument('--theta2', type=float, default=1.0)
    parser.add_argument('--theta3', type=float, default=0.01)
    parser.add_argument('--num_steps', type=int, default=20000)
    args = parser.parse_args()

    sim = LotkaVolterraSimulator(X0=args.X0, Y0=args.Y0,
            theta0=args.theta0, theta1=args.theta1,
            theta2=args.theta2, theta3=args.theta3)

    batches = sim.simulate_tasks(args.num_batches, args.batch_size,
            num_steps=args.num_steps)

    root = os.path.join(datasets_path, 'lotka_volterra')
    if not os.path.isdir(root):
        os.makedirs(root)

    torch.save(batches, os.path.join(root, f'{args.filename}.tar'))

    fig, axes = plt.subplots(1, 4, figsize=(16,4))
    for i, ax in enumerate(axes.flatten()):
        ax.scatter(batches[0].x[i,:,0], batches[0].y[i,:,0])
        ax.scatter(batches[0].x[i,:,0], batches[0].y[i,:,1])
    plt.show()

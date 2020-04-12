import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
import torch
from torch.autograd import grad

from model import *

def run_exp(index, logdir):
    # hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_x = 100 # number of training data
    lr = 0.005 # learning rate
    maxiter = 500 # number of iterations
    noise_std = 0.01 # noise std

    # model
    model = Basin().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr)

    # data
    X_np = np.matmul(np.random.randn(num_x, 2) * np.array([math.sqrt(10), 1]), model.K) * 1e-2
    X = torch.FloatTensor(X_np).to(device)

    # generate path
    weights_dic = {'gd':None, 'gld':None, 'diag':None, 'cov':None, 'hess':None, 'hessMax':None}
    
    weights_dic['gd'] = train_gd(X, model, optimizer, maxiter)
    weights_dic['gld'] = train_gld(X, model, optimizer, maxiter, noise_std)
    weights_dic['diag'] = train_diag(X, model, optimizer, maxiter, noise_std)
    weights_dic['cov'] = train_cov(X, model, optimizer, maxiter, noise_std)
    weights_dic['hess'] = train_hess(X, model, optimizer, maxiter, noise_std)
    weights_dic['hessMax'] = train_hessMax(X, model, optimizer, maxiter, noise_std)
    
    np.savez(os.path.join(logdir, 'weight.npz'), X=X_np,
            gd=weights_dic['gd'], 
            gld=weights_dic['gld'],
            diag=weights_dic['diag'],
            cov=weights_dic['cov'], 
            hess=weights_dic['hess'],
            hessMax=weights_dic['hessMax'])

    # count success case
    success_dic = {}
    for key in weights_dic.keys():
        if weights_dic[key] is not None:
            if weights_dic[key][-1, 0] < -0.5 and weights_dic[key][-1, 1] < -0.5:
                success_dic[key] = 1
            else:
                success_dic[key] = 0

    # plot
    w1 = np.arange(-1.4, 1.6, 0.02)
    w2 = np.arange(-1.4, 1.6, 0.02)
    W1, W2 = np.meshgrid(w1, w2)
    s = np.array([np.mean(model.show(w1, w2, X_np)) for w1, w2 in zip(np.ravel(W1), np.ravel(W2))])
    S = s.reshape(W1.shape)

    fig = plt.figure(index+1)
    c = plt.contour(W1, W2, S, 8)
    plt.clabel(c, inline=True, fontsize=10)
    plt.xlabel('w1', fontsize=18)
    plt.ylabel('w2', fontsize=18)
    # plt.savefig(os.path.join(logdir, 'contour.pdf'))

    labels = {'gd':'GD', 'gld':'GLD const', 'diag':'GLD diag', 'cov':'GLD leading', 'hess':'GLD Hessian', 'hessMax':'GLD 1st eigven(H)'}
    colors = {'gd':'b', 'gld':'g', 'diag':'c', 'cov':'m', 'hess':'y', 'hessMax':'dimgray'}
    marks = {'gd':'_', 'gld':'*', 'diag':'s', 'cov':'^', 'hess':'|', 'hessMax':'v'}
    mask = np.linspace(0, maxiter-1, 100, dtype=int)
    for key in weights_dic.keys():
        if weights_dic[key][i,0] < -1.4 or weights_dic[key][i,1] < -1.4:
                weights_dic[key][i,0] = 1
                weights_dic[key][i,1] = 1    
        plt.scatter(weights_dic[key][mask][:,0], weights_dic[key][mask][:,1], c=colors[key], marker=marks[key], label=labels[key], s=20)
    plt.legend(loc='lower right', markerscale=1, fontsize=14)
    plt.savefig(os.path.join(logdir, 'trajectory.pdf'))
    
    return success_dic



if __name__ == '__main__':
    success_count = {'gd':0, 'gld':0, 'diag':0, 'cov':0, 'hess':0, 'hessMax':0}
    for i in range(100):
        logdir = os.path.join('./logs', 'exp_'+str(i))
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        success_dic = run_exp(i, logdir)
        for key in success_dic.keys():
            success_count[key] += success_dic[key]
        print('\n\n=== in exp:', i, 'success:', success_count, '===\n\n')
    
    np.save(os.path.join(logdir, 'count.npy'), success_count)

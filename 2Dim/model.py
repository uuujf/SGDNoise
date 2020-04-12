import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import grad

class Basin(nn.Module):
    def __init__(self):
        super(Basin, self).__init__()
        self.n_weights = 2
        self.weight = torch.nn.Parameter(torch.ones(self.n_weights), requires_grad=True)

        self.theta = math.pi / 4
        self.K1 = math.cos(self.theta)
        self.K2 = math.sin(self.theta)
        self.K = np.array([[self.K1, -self.K2], [self.K2, self.K1]])

    def initialize(self):
        self.weight.data = torch.ones_like(self.weight.data)

    def forward(self, x):
        x1, y1 = self.weight[0]-x[:,0]-1, self.weight[1]-x[:,1]-1
        x2, y2 = self.weight[0]-x[:,0]+1, self.weight[1]-x[:,1]+1
        out = torch.min(10*(self.K1*x1 - self.K2*y1)**2 + 100*(self.K1*x1 + self. K2*y1)**2, (x2)**2 + (y2)**2)
        return out

    def show(self, w0, w1, x):
        x1, y1 = w0-x[:,0]-1, w1-x[:,1]-1
        x2, y2 = w0-x[:,0]+1, w1-x[:,1]+1
        out = np.minimum(10*(self.K1*x1 - self.K2*y1)**2 + 100*(self.K1*x1 + self.K2*y1)**2, (x2)**2 + (y2)**2)
        return out


def eval_Hess(X, model):
    loss = model(X).mean()
    g = grad(loss, model.parameters(), retain_graph=True, create_graph=True)[0]
    hessian = []
    for gi in g:
        gg = grad(gi, model.parameters(), retain_graph=True)[0]
        hessian.append(gg.data.clone())
    hessian = torch.stack(hessian, dim=0)
    return hessian

def eval_Cov(X, model):
    grads = []
    for j in range(len(X)):
        X_b = X[j].view(-1, 2)
        loss = model(X_b).mean()
        loss.backward()
        grads.append(model.weight.grad.data.clone())
    grads = torch.stack(grads, dim=0)
    grad_mean = torch.mean(grads, dim=0)
    grads = grads - grad_mean.view(1,-1).expand_as(grads)
    covariance = torch.mm(torch.t(grads), grads) / len(X)
    return covariance


def train_gd(X, model, optimizer, maxiter):
    model.initialize()
    weights = []
    print('======GD======')
    for i in range(maxiter):
        loss = model(X).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        weights.append(model.weight.data.cpu().clone().numpy())
        if i % 100 == 0:
            print('Step:', i, 'Loss:', loss.item(), 'Location:', model.weight.data)
    return np.stack(weights)

def train_gld(X, model, optimizer, maxiter, noise_std):
    model.initialize()
    weights = []
    print('======GLD======')
    std = math.sqrt(noise_std / model.n_weights)
    for i in range(maxiter):
        loss = model(X).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.weight.data += torch.randn(model.weight.size()).to(model.weight.device) * std
        weights.append(model.weight.data.cpu().clone().numpy())
        if i % 100 == 0:
            print('Step:', i, 'Loss:', loss.item(), 'Location:', model.weight.data)
    return np.stack(weights)

def train_diag(X, model, optimizer, maxiter, noise_std):
    model.initialize()
    weights = []
    print('======GLD Diag======')
    for i in range(maxiter):
        covariance = eval_Cov(X, model)
        std = (covariance.diag() / covariance.trace() * noise_std).sqrt()

        loss = model(X).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.weight.data += torch.randn(model.n_weights).to(model.weight.device) * std
        weights.append(model.weight.data.cpu().clone().numpy())
        if i % 100 == 0:
            print('Step:', i, 'Loss:', loss.item(), 'Location:', model.weight.data)
    return np.stack(weights)

def train_cov(X, model, optimizer, maxiter, noise_std):
    model.initialize()
    weights = []
    print('======GLD Leading======')
    for i in range(maxiter):
        covariance = eval_Cov(X, model)
        covariance = covariance / covariance.trace() * noise_std
        e, v = torch.symeig(covariance, eigenvectors=True)

        loss = model(X).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.weight.data += torch.matmul(v, torch.randn(model.n_weights).to(model.weight.device) * e.sqrt())
        weights.append(model.weight.data.cpu().clone().numpy())
        if i % 100 == 0:
            print('Step:', i, 'Loss:', loss.item(), 'Location:', model.weight.data)
    return np.stack(weights)

def train_hess(X, model, optimizer, maxiter, noise_std):
    model.initialize()
    weights = []
    print('======GLD Hess======')
    for i in range(maxiter):
        hessian = eval_Hess(X, model)
        hessian = hessian / hessian.trace() * noise_std
        e, v = torch.symeig(hessian, eigenvectors=True)

        loss = model(X).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.weight.data += torch.matmul(v, torch.randn(model.n_weights).to(model.weight.device) * e.sqrt())
        weights.append(model.weight.data.cpu().clone().numpy())
        if i % 100 == 0:
            print('Step:', i, 'Loss:', loss.item(), 'Location:', model.weight.data)
    return np.stack(weights)

def train_hessMax(X, model, optimizer, maxiter, noise_std):
    model.initialize()
    weights = []
    print('======GLD HessMax======')
    for i in range(maxiter):
        hessian = eval_Hess(X, model)
        e, v = torch.symeig(hessian, eigenvectors=True)
        std = math.sqrt(noise_std)

        loss = model(X).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.weight.data += v[:,-1] * torch.randn(1).to(model.weight.device) * std
        weights.append(model.weight.data.cpu().clone().numpy())
        if i % 100 == 0:
            print('Step:', i, 'Loss:', loss.item(), 'Location:', model.weight.data[0], model.weight.data[1],
                    'v:', v[0,0], v[0,1], v[1,0], v[1,1], 'e:', e[0], e[1])
    return np.stack(weights)

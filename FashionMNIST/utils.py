import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh
import tensorflow as tf

def cross_entropy_with_softmax(output, target, epsilon=1e-6):
    prob = tf.softmax(output, axis=1)
    loss = tf.reduce_sum(target * tf.log(prob + epsilon), axis=1)
    loss = - tf.reduce_min(loss)
    return loss

class Loader(object):
    def __init__(self, dataset, batch_size, shuffle):
        super(Loader, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self):
        idx = np.arange(self.dataset.n_examples)
        if self.shuffle:
            np.random.shuffle(idx)

        data_loader = []
        i = 0
        while (i + self.batch_size <= self.dataset.n_examples):
            data_loader.append((self.dataset.X[idx[i:i+self.batch_size]],
                                self.dataset.Y[idx[i:i+self.batch_size]]))
            i += self.batch_size

        #if i < self.dataset.n_examples:
        #    data_loader.append((self.dataset.images[idx[i:self.dataset.n_examples]],
        #                        self.dataset.labels[idx[i:self.dataset.n_examples]]))
        return data_loader

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def l0_norm(holder, epsilon=1e-9):
    mask = tf.less(tf.abs(holder), epsilon*tf.ones_like(holder))
    mask = tf.cast(mask, tf.int32)
    return tf.reduce_sum(mask)

def make_dir(full_dir):
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    return full_dir

def save_ckpt_with_iter(dict, dir, iteration):
    torch.save(dict, os.path.join(dir, 'iter_'+str(iteration)+'.pth.tar'))
    return iteration
    

def train(sess, data_loader, model, lr, keep_prob=1.0, noise=None):
    losses = AverageMeter()
    accs = AverageMeter()

    if noise is None:
        noise = np.zeros(model.n_weights)

    for i, (x, y) in enumerate(data_loader):
        _, loss, acc = sess.run([model.optimize_opt, model.loss, model.accuracy],
                                feed_dict={
                                    model.x: x,
                                    model.y: y,
                                    model.lr: lr,
                                    model.is_train: True,
                                    model.keep_prob: keep_prob,
                                    model.grad_noise: noise})
        losses.update(loss, len(x))
        accs.update(acc, len(x))
    return losses.avg, accs.avg


def validate(sess, val_loader, model):
    losses = AverageMeter()
    accs = AverageMeter()

    for i, (x, y) in enumerate(val_loader):
        loss, acc = sess.run([model.loss, model.accuracy],
                                feed_dict={
                                    model.x: x,
                                    model.y: y,
                                    model.is_train: False,
                                    model.keep_prob: 1.0
                                })
        losses.update(loss, len(x))
        accs.update(acc, len(x))

    return losses.avg, accs.avg

def eval_Hess(sess, whole_loader, model):
    x, y = whole_loader[0]
    hessian = []
    for i in range(model.n_weights):
        v = np.zeros(model.n_weights); v[i] = 1
        gi = sess.run(model.gg, feed_dict={
            model.x: x,
            model.y: y,
            model.v: v,
            model.is_train: False,
            model.keep_prob: 1.0})
        hessian.append(gi)
    hessian = np.stack(hessian)
    return hessian

def eval_Cov(sess, one_loader, model):
    n_batches = len(one_loader)
    grads = []
    for i, (x, y) in enumerate(one_loader):
        if i < n_batches:
            grad = sess.run(model.gradient, feed_dict={
                    model.x: x,
                    model.y: y,
                    model.is_train: False,
                    model.keep_prob: 1.0})
            grads.append(grad)
    grads = np.stack(grads)
    grad_mean = np.mean(grads, axis=0)
    grads = grads - np.tile(grad_mean, (n_batches,1))
    covariance = np.matmul(grads.T, grads)/(n_batches-1)
    return covariance, grad_mean

def eval_curve(sess, val_loader, model, epoch, dir):
    x, y = val_loader[0]
    out = sess.run(model.out, feed_dict={
                    model.x: x,
                    model.y: y,
                    model.is_train: False,
                    model.keep_prob: 1.0})

    fig = plt.figure(1)
    plt.plot(x, out, 'r--')
    plt.plot(x, y, 'b')
    plt.savefig(os.path.join(dir,str(epoch)))
    plt.clf()
    return True
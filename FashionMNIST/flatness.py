import os
import argparse
import math
import numpy as np
import tensorflow as tf

from mnist import MNIST
from model import ConvYu
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', default='/Users/wjf/datasets/fashion_tf_1k_cor', type=str)
parser.add_argument('--start', default=0, type=int, help='start iter (k)')
parser.add_argument('--end', default=16, type=int, help='end iter (k)')
parser.add_argument('--std', default=1e-2, type=float, help='noise std')
parser.add_argument('--n-iter', default=1000, type=int, help='num of eval iteration')
parser.add_argument('--ckptdir', default='logs/gd', type=str, help='relative ckpts dir')
parser.add_argument('--logdir', default='flatness/gd', type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    # data loader
    train_set = MNIST(os.path.join(args.datadir, 'train.npz'))
    val_set = MNIST(os.path.join(args.datadir, 'val.npz'))
    val_loader = Loader(val_set, batch_size=500, shuffle=False)

    # model
    model = ConvYu()

    # summary
    _loss0 = tf.placeholder(tf.float32)
    _acc0 = tf.placeholder(tf.float32)
    _loss = tf.placeholder(tf.float32)
    _acc = tf.placeholder(tf.float32)
    _dloss = tf.placeholder(tf.float32)
    _dacc = tf.placeholder(tf.float32)

    train_summary_list = [tf.summary.scalar('loss0/train', _loss0),
                          tf.summary.scalar('acc0/train', _acc0),
                          tf.summary.scalar('loss/train', _loss),
                          tf.summary.scalar('acc/train', _acc),
                          tf.summary.scalar('delta_loss/train', _dloss),
                          tf.summary.scalar('delta_acc/train', _dacc)]
    train_summary_merged = tf.summary.merge(train_summary_list)
    val_summary_list = [tf.summary.scalar('loss0/val', _loss0),
                        tf.summary.scalar('acc0/val', _acc0),
                        tf.summary.scalar('loss/val', _loss),
                        tf.summary.scalar('acc/val', _acc),
                        tf.summary.scalar('delta_loss/val', _dloss),
                        tf.summary.scalar('delta_acc/val', _dacc)]
    val_summary_merged = tf.summary.merge(val_summary_list)

    # eval flatness
    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        # initialize
        sess.run(tf.global_variables_initializer())

        # writer
        writer = tf.summary.FileWriter(make_dir(args.logdir), sess.graph)

        ckpts = {}
        for i in range(args.start, args.end):
            ckpt_file = os.path.join(args.ckptdir, 'model-'+str(i*1000))
            saver.restore(sess, ckpt_file)
            print("=> loading checkpoint '{}'".format(ckpt_file))
            ckpts[i] = sess.run(model.weight)

        for i in range(args.start, args.end):
            lossT0, accT0 = sess.run([model.eval_loss, model.eval_acc], feed_dict={
                                    model.x: train_set.X,
                                    model.y: train_set.Y,
                                    model.w: ckpts[i]})
            lossV0, accV0 = sess.run([model.eval_loss, model.eval_acc], feed_dict={
                                    model.x: val_set.X,
                                    model.y: val_set.Y,
                                    model.w: ckpts[i]})
            lossesT, accsT = AverageMeter(), AverageMeter()
            lossesV, accsV = AverageMeter(), AverageMeter()
            dlossesT, daccsT = AverageMeter(), AverageMeter()
            dlossesV, daccsV = AverageMeter(), AverageMeter()

            for j in range(args.n_iter):
                noise = np.random.randn(model.n_weights) * args.std
                lossT, accT = sess.run([model.eval_loss, model.eval_acc], feed_dict={
                                        model.x: train_set.X,
                                        model.y: train_set.Y,
                                        model.w: ckpts[i]+noise})
                lossV, accV = sess.run([model.eval_loss, model.eval_acc], feed_dict={
                                        model.x: val_set.X,
                                        model.y: val_set.Y,
                                        model.w: ckpts[i]+noise})
                dlossesT.update(abs(lossT-lossT0)); daccsT.update(abs(accT-accT0))
                dlossesV.update(abs(lossV-lossV0)); daccsV.update(abs(accV-accV0))
                lossesT.update(lossT); accsT.update(accT)
                lossesV.update(lossV); accsV.update(accV)

            summary = sess.run(train_summary_merged, feed_dict={
                                _loss0:lossT0,
                                _acc0:accT0,
                                _loss:lossesT.avg,
                                _acc:accsT.avg,
                                _dloss:dlossesT.avg,
                                _dacc:daccsT.avg})
            writer.add_summary(summary, i)
            summary = sess.run(val_summary_merged, feed_dict={
                                _loss0:lossV0,
                                _acc0:accV0,
                                _loss:lossesV.avg,
                                _acc:accsV.avg,
                                _dloss:dlossesV.avg,
                                _dacc:daccsV.avg})
            writer.add_summary(summary, i)
            print('model: {iteration}   noise: {noise:.4f}   DeltaLossTrain {lossT:.6f}   DeltaAccVal {accV:.6f}'
              .format(iteration=i, noise=args.std, lossT=dlossesT.avg, accV=daccsV.avg))


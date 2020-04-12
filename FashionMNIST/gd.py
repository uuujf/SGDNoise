import os
import argparse
import math
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh
import tensorflow as tf

from mnist import MNIST
from model import ConvYu
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--maxiter', default=20000, type=int)
parser.add_argument('--keep-prob',  default=1.0, type=float)
parser.add_argument('--lr',  default=0.1, type=float)
parser.add_argument('--datadir', default='/Users/wjf/datasets/fashion_tf_1k_cor', type=str)
parser.add_argument('--logdir', default='logs/gd', type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    # data loader
    train_set = MNIST(os.path.join(args.datadir, 'train.npz'))
    val_set = MNIST(os.path.join(args.datadir, 'val.npz'))
    val_loader = Loader(val_set, batch_size=500, shuffle=False)

    # model
    model = ConvYu()

    # summary
    _loss = tf.placeholder(tf.float32)
    _acc = tf.placeholder(tf.float32)

    train_summary_list = [tf.summary.scalar('loss/train', _loss),
                          tf.summary.scalar('acc/train', _acc)]
    train_summary_merged = tf.summary.merge(train_summary_list)
    val_summary_list = [tf.summary.scalar('loss/val', _loss),
                        tf.summary.scalar('acc/val', _acc)]
    val_summary_merged = tf.summary.merge(val_summary_list)

    # prepare
    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        # initialize
        sess.run(tf.global_variables_initializer())

        # resume
        if args.resume is not None:
            saver.restore(sess, args.resume)
            print("Model restored.")

        # writer
        writer = tf.summary.FileWriter(make_dir(args.logdir), sess.graph)

        # training and eval
        for epoch in range(args.maxiter):
            # training step
            data_tuple = (train_set.X, train_set.Y)
            loss, acc = train(sess, [data_tuple], model, args.lr, args.keep_prob)
            summary = sess.run(train_summary_merged, feed_dict={_loss:loss, _acc:acc})
            writer.add_summary(summary, epoch)
            print('Epoch: {:}    loss: {:.6f}    acc: {:.2f}    In Training'.format(epoch, loss, acc))

            if epoch % 1 == 0:
                # validate step
                loss, acc = validate(sess, val_loader(), model)
                summary = sess.run(val_summary_merged, feed_dict={_loss:loss, _acc:acc})
                writer.add_summary(summary, epoch)
                print('Epoch: {:}    loss: {:.6f}    acc: {:.2f}    In Validation'.format(epoch, loss, acc))

            # save ckpt
            if epoch % 1 == 0:
                saver.save(sess, os.path.join(args.logdir, 'model'), epoch)


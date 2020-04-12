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
parser.add_argument('--lr',  default=0.07, type=float)
parser.add_argument('--lead', default=20, type=int)
parser.add_argument('--batch-size', default=20, type=int)
parser.add_argument('--frequence', default=10, type=int)
parser.add_argument('--datadir', default='/Users/wjf/datasets/fashion_tf_1k_cor', type=str)
parser.add_argument('--logdir', default='logs/gld_lead', type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    # data loader
    train_set = MNIST(os.path.join(args.datadir, 'train.npz'))
    val_set = MNIST(os.path.join(args.datadir, 'val.npz'))
    val_loader = Loader(val_set, batch_size=500, shuffle=False)
    one_loader = Loader(train_set, batch_size=args.batch_size, shuffle=True)

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

    _ecov = tf.placeholder(tf.float32, [args.lead])
    _vcov = tf.placeholder(tf.float32, [model.n_weights, args.lead])
    _l2norm_cov = tf.placeholder(tf.float32)
    _trnorm_cov = tf.placeholder(tf.float32)

    matrix_summary_list = [tf.summary.scalar('matrix/l2_norm/cov', _l2norm_cov),
                           tf.summary.scalar('matrix/tr_norm/cov', _trnorm_cov),
                           tf.summary.scalar('leadingPer', tf.reduce_sum(_ecov)/tf.square(_trnorm_cov))]
    for i in range(args.lead):
        matrix_summary_list.append(tf.summary.scalar('eigenValue/lead'+str(i+1)+'/ecov', _ecov[i]))
    matrix_summary_merged = tf.summary.merge(matrix_summary_list)

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
            data_tuple = (train_set.X, train_set.Y)
            noise = np.zeros(model.n_weights)
            if epoch % args.frequence == 0:
                # eval matrix
                covariance, _ = eval_Cov(sess, one_loader(), model)
                l2norm_cov = np.linalg.norm(covariance, ord='fro')
                trnorm_cov = np.sqrt(covariance.trace())
                ecov, vcov = eigsh(covariance, args.lead, which='LM')
                summary = sess.run(matrix_summary_merged, feed_dict={
                    _l2norm_cov: l2norm_cov,
                    _trnorm_cov: trnorm_cov,
                    _ecov: ecov})
                writer.add_summary(summary, epoch)
                noise = np.matmul(vcov, np.random.normal(0, np.sqrt(np.abs(ecov))))

            # training step
            loss, acc = train(sess, [data_tuple], model, args.lr, args.keep_prob, noise)
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
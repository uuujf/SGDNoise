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
parser.add_argument('--frequence', default=1, type=int)
parser.add_argument('--datadir', default='/Users/wjf/datasets/fashion_tf_1k_cor', type=str)
parser.add_argument('--logdir', default='logs/gld_1Hess', type=str)

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

    _grad = tf.placeholder(tf.float32, [model.n_weights])
    _ehess = tf.placeholder(tf.float32, [args.lead])
    _vhess = tf.placeholder(tf.float32, [model.n_weights, args.lead])
    _ecov = tf.placeholder(tf.float32, [args.lead])
    _vcov = tf.placeholder(tf.float32, [model.n_weights, args.lead])
    _l2norm_hess = tf.placeholder(tf.float32)
    _l2norm_cov = tf.placeholder(tf.float32)
    _trnorm_hess = tf.placeholder(tf.float32)
    _trnorm_cov = tf.placeholder(tf.float32)
    _vec_dis = tf.norm(_vcov-tf.matmul(_vhess, tf.matmul(tf.transpose(_vhess), _vcov)), ord='euclidean')
    _vTv = tf.matmul(tf.transpose(_vhess), _vcov)

    grad_summary_list = [tf.summary.scalar('grad/l0_norm', l0_norm(_grad)),
                         tf.summary.scalar('grad/l2_norm', tf.norm(_grad, ord=2))]
    matrix_summary_list = [tf.summary.scalar('matrix/l2_norm/hess', _l2norm_hess),
                           tf.summary.scalar('matrix/l2_norm/cov', _l2norm_cov),
                           tf.summary.scalar('matrix/tr_norm/hess', _trnorm_hess),
                           tf.summary.scalar('matrix/tr_norm/cov', _trnorm_cov),
                           tf.summary.scalar('matrix/projection_fro', _vec_dis)]
    for i in range(args.lead):
        matrix_summary_list.append(tf.summary.scalar('eigenValue/lead'+str(i+1)+'/ehess', _ehess[i]))
        matrix_summary_list.append(tf.summary.scalar('eigenValue/lead'+str(i+1)+'/ecov', _ecov[i]))
        matrix_summary_list.append(tf.summary.scalar('cosin_angle/lead'+str(i+1), tf.abs(_vTv[i, i])))
    matrix_image_summary_list = [tf.summary.image('vTv', tf.reshape(tf.abs(_vTv), [1,args.lead,args.lead,1]), max_outputs=1)]
    matrix_summary_merged = tf.summary.merge(grad_summary_list + matrix_summary_list + matrix_image_summary_list)

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
            # update matrix every ** iters
            if epoch % args.frequence == 0:
                # power method
                vhess = np.random.rand(model.n_weights)
                for i in range(30):
                    vhess = sess.run(model.gg, feed_dict={model.x:train_set.X, model.y:train_set.Y, model.v:vhess, model.is_train:False, model.keep_prob:1.0})
                    vhess = vhess / np.linalg.norm(vhess, ord=2)
                covariance, grad_mean = eval_Cov(sess, one_loader(), model)
                ehessBar = np.trace(covariance)
            noise = vhess * np.random.normal(0, math.sqrt(ehessBar))

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
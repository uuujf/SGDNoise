import os
from datetime import datetime
import math
import argparse
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=123, type=int)
parser.add_argument('--iter-per-epoch', default=10, type=int)
parser.add_argument('--epoch', default=31, type=int,)
parser.add_argument('--lr-decay-epoch', default=21, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--batch-size', default=10, type=int)
parser.add_argument('--delta', default=1e-3, type=float)
parser.add_argument('--inputnodes', default=2, type=int)
parser.add_argument('--hiddennodes', default=100, type=int)
parser.add_argument('--numX', default=1000, type=int)
parser.add_argument('--name', default='./logs', type=str)

def main():
    args = parser.parse_args()
    print(args)
    log_dir = args.name+'/numX'+str(args.numX)+'_hidden'+str(args.hiddennodes)+'_seed'+str(args.seed)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
        f.write(str(args))
    # set random seed
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    # dataset
    X_train = np.random.uniform(-2, 2, size=(args.numX, args.inputnodes))
    Y_train = np.ones(args.numX)
    Y_train[X_train[:,0]<X_train[:,1]] = 0.0
    Y_train[X_train[:,0]>=X_train[:,1]] = 1.0
    print(Y_train.sum()/len(Y_train))

    # build graph
    net = OneHiddenLayer(args.inputnodes, args.hiddennodes, args.delta)
    net.build_graph()
    _trace, _tracebar = tf.placeholder(tf.float32,name='trace'), tf.placeholder(tf.float32,name='trace_bar')
    trace_summary = tf.summary.merge([
            tf.summary.scalar('trace/HSigma', _trace),
            tf.summary.scalar('trace/HSigmaBar', _tracebar), ])

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(tf.global_variables_initializer(), feed_dict={net.lr:args.lr})
        for ep in range(args.epoch):
            if ep < args.lr_decay_epoch:
                decayed_lr = args.lr
            else:
                decayed_lr = args.lr * (args.epoch-ep)/float(args.epoch-args.lr_decay_epoch)

            for _ in range(args.iter_per_epoch):
                _ = sess.run(net.train_op, feed_dict={net.x:X_train, net.y:Y_train, net.lr:decayed_lr})
            loss_train, acc_train, summary_train = sess.run([net.loss, net.accuracy, net.summary], feed_dict={net.x:X_train, net.y:Y_train})
            writer.add_summary(summary_train, ep)
            print('epoch', ep, 'loss', loss_train, 'acc_train', acc_train)

            hess = eval_Hess(sess, (X_train,Y_train), net)
            batch_tuples = []
            for _ in range(100):
                mask = np.random.choice(args.numX, args.batch_size, False)
                batch_tuples.append((X_train[mask], Y_train[mask]))
            # from IPython import embed;embed()
            cov, _ = eval_Cov(sess, batch_tuples, net)
            trace = np.matmul(hess, cov).trace()
            tracebar = hess.trace()*cov.trace()/net.n_weights
            writer.add_summary(sess.run(trace_summary,feed_dict={_trace:trace,_tracebar:tracebar}), ep)

class OneHiddenLayer(object):
    def __init__(self, inputnodes=2, hiddennodes=100, delta=0.01):
        super(OneHiddenLayer, self).__init__()
        self.reuse = {}
        self.inputnodes = inputnodes
        self.hiddennodes = hiddennodes
        self.delta = delta

    def classifier(self, x, name='one_hidden_layer'):
        if name in self.reuse.keys():
            reuse = self.reuse[name]
        else:
            self.reuse[name] = True
            reuse = False

        with tf.variable_scope(name, reuse=reuse) as scope:
            self.n_weights = self.hiddennodes*(self.inputnodes+1)
            self.weight = tf.get_variable(name='weight', shape=[self.n_weights], dtype=tf.float32)
            w = tf.reshape(self.weight[:self.hiddennodes*self.inputnodes], [self.inputnodes,self.hiddennodes])
            b = tf.reshape(self.weight[self.hiddennodes*self.inputnodes:], [self.hiddennodes])
            fixed = tf.concat([tf.ones((self.hiddennodes/2,1)), -tf.ones((self.hiddennodes/2,1))], axis=0)

            x = tf.matmul(x, w) + b
            x = tf.nn.relu(x)
            x = tf.matmul(x, fixed)
            x = tf.maximum(tf.minimum(x, 1-self.delta), self.delta)
            # x = tf.nn.sigmoid(x)
            x = tf.reshape(x, [-1])
            return x

    def build_graph(self):
        self.lr = tf.placeholder(tf.float32, [], name='lr')
        self.x = tf.placeholder(tf.float32, [None, self.inputnodes], name='x')
        self.y = tf.placeholder(tf.float32, [None], name='y')

        out = self.classifier(self.x, 'one_hidden_layer')
        self.loss = tf.reduce_mean(tf.square(out-self.y))
        # self.loss = -tf.reduce_mean(self.y*tf.log(out+1e-8) + (1-self.y)*tf.log(1-out+1e-8))

        pred = tf.cast(out>0.5, tf.float32)
        correct_prediction = tf.equal(self.y, pred)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.summary = tf.summary.merge([
            tf.summary.scalar('loss/train', self.loss),
            tf.summary.scalar('acc/train', self.accuracy), ])

        weight_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='one_hidden_layer')
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=weight_list)
        # self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, var_list=weight_list)

        self.gradient = tf.gradients(self.loss, self.weight)[0]
        self.v = tf.placeholder(tf.float32, [self.n_weights], name='vector')
        self.gg = tf.gradients(self.gradient*self.v, self.weight)[0]

def eval_Hess(sess, data_tuple, model):
    x, y = data_tuple
    hessian = []
    for i in range(model.n_weights):
        v = np.zeros(model.n_weights); v[i] = 1
        gi = sess.run(model.gg, feed_dict={model.x:x, model.y:y, model.v:v})
        hessian.append(gi)
    hessian = np.stack(hessian)
    return hessian

def eval_Cov(sess, batch_tuples, model):
    grads = []
    for (x, y) in batch_tuples:
        grad = sess.run(model.gradient, feed_dict={model.x:x, model.y:y})
        grads.append(grad)
    grads = np.stack(grads)
    grad_mean = np.mean(grads, axis=0)
    grads = grads - np.tile(grad_mean, (len(batch_tuples),1))
    covariance = np.matmul(grads.T, grads)/(len(batch_tuples)-1)
    return covariance, grad_mean

if __name__ == '__main__':
    main()

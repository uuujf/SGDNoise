import os
from six.moves import reduce
import numpy as np
import tensorflow as tf

class ConvYu(object):
    def __init__(self):
        super(ConvYu, self).__init__()
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.lr = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32)
        self.reuse = {}
        self._build_model()

    def cls(self, name, X):
        if name in self.reuse.keys():
            reuse = self.reuse[name]
        else:
            self.reuse[name] = True
            reuse = False

        with tf.variable_scope(name, reuse=reuse) as scope:
            self.n_weights = 11330
            self.weight = tf.get_variable(name='weight',
                                          shape=[self.n_weights],
                                          dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.02))
            conv1_W = tf.reshape(self.weight[0:250], [5,5,1,10])
            conv1_b = tf.reshape(self.weight[250:260], [10])
            conv2_W = tf.reshape(self.weight[260:2760], [5,5,10,10])
            conv2_b = tf.reshape(self.weight[2760:2770], [10])
            fc1_W = tf.reshape(self.weight[2770:10770], [160,50])
            fc1_b = tf.reshape(self.weight[10770:10820], [50])
            fc2_W = tf.reshape(self.weight[10820:11320], [50,10])
            fc2_b = tf.reshape(self.weight[11320:11330], [10])

            out = X
            out = tf.nn.conv2d(out, conv1_W, [1,1,1,1], 'VALID') + conv1_b
            out = tf.nn.max_pool(out, [1,2,2,1], [1,2,2,1], 'VALID')
            out = tf.nn.relu(out)

            out = tf.nn.conv2d(out, conv2_W, [1,1,1,1], 'VALID') + conv2_b
            out = tf.nn.max_pool(out, [1,2,2,1], [1,2,2,1], 'VALID')
            out = tf.nn.relu(out)

            flat_dim = reduce(lambda x,y: x*y, out.get_shape().as_list()[1:])
            out = tf.reshape(out, [-1, flat_dim])
            out = tf.matmul(out, fc1_W) + fc1_b
            out = tf.nn.relu(out)
            out = tf.matmul(out, fc2_W) + fc2_b
            return out

    def eval_loss_acc(self, X, Y, W):
        conv1_W = tf.reshape(W[0:250], [5,5,1,10])
        conv1_b = tf.reshape(W[250:260], [10])
        conv2_W = tf.reshape(W[260:2760], [5,5,10,10])
        conv2_b = tf.reshape(W[2760:2770], [10])
        fc1_W = tf.reshape(W[2770:10770], [160,50])
        fc1_b = tf.reshape(W[10770:10820], [50])
        fc2_W = tf.reshape(W[10820:11320], [50,10])
        fc2_b = tf.reshape(W[11320:11330], [10])

        out = X
        out = tf.nn.conv2d(out, conv1_W, [1,1,1,1], 'VALID') + conv1_b
        out = tf.nn.max_pool(out, [1,2,2,1], [1,2,2,1], 'VALID')
        out = tf.nn.relu(out)

        out = tf.nn.conv2d(out, conv2_W, [1,1,1,1], 'VALID') + conv2_b
        out = tf.nn.max_pool(out, [1,2,2,1], [1,2,2,1], 'VALID')
        out = tf.nn.relu(out)

        flat_dim = reduce(lambda x,y: x*y, out.get_shape().as_list()[1:])
        out = tf.reshape(out, [-1, flat_dim])
        out = tf.matmul(out, fc1_W) + fc1_b
        out = tf.nn.relu(out)
        out = tf.matmul(out, fc2_W) + fc2_b

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=out))
        acc = tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(out, 1)), tf.float32)
        acc = tf.reduce_mean(acc) * 100.0
        return loss, acc

    def _accuracy(self, labels, logits, mark=None):
        acc = tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1)), tf.float32)
        if mark == None:
            acc = tf.reduce_mean(acc)
            return acc*100.0
        else:
            numT = tf.maximum(tf.reduce_sum(mark), 1)
            numF = tf.maximum(tf.reduce_sum(1-mark), 1)
            accT = tf.reduce_sum(acc*mark) / numT
            accF = tf.reduce_sum(acc*(1-mark)) / numF
        return accT*100.0, accF*100.0

    def switch_opt(self, opt_type='RMSProp'):
        if opt_type == 'RMSProp':
            self._optimizer = tf.train.RMSPropOptimizer(self.lr)
            print('use rmsprop')
        elif opt_type == 'Momentum':
            self._optimizer = tf.train.MomentumOptimizer(self.lr, momentum=0.9, use_nesterov=False)
            print('use momentum')
        else:
            print('Cannot recongnize opt type. Opt is not change.')

    def _build_model(self):
        self.cls_name = 'ConvSmallYu'
        self.out = self.cls(self.cls_name, self.x)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.out))
        self.accuracy = self._accuracy(labels=self.y, logits=self.out)

        self._optimizer = tf.train.MomentumOptimizer(self.lr, momentum=0, use_nesterov=False)
        # self._optimizer = tf.train.RMSPropOptimizer(self.lr)
        self.weight_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.cls_name)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.grad_tuple = self._optimizer.compute_gradients(loss=self.loss, var_list=self.weight_list)
            # self.optimize_opt = self._optimizer.apply_gradients(self.grad_tuple)
            self.grad_noise = tf.placeholder(tf.float32, [self.n_weights])
            self.optimize_opt = self._optimizer.apply_gradients([(self.grad_tuple[0][0]+self.grad_noise, self.grad_tuple[0][1])])

        self.gradient = self.grad_tuple[0][0]
        self.v = tf.placeholder(tf.float32, [self.n_weights])
        self.gg = tf.gradients(self.gradient*self.v, self.weight_list)[0]

        self.w = tf.placeholder(tf.float32, [self.n_weights])
        self.eval_loss, self.eval_acc = self.eval_loss_acc(self.x, self.y, self.w)
        return True

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    model = ConvSmallYu()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    x = np.ones((2,28,28,1))
    y = np.zeros((2,10))
    v = np.zeros(model.n_weights); v[1]=1
    out = sess.run(model.out, feed_dict={model.x:x, model.keep_prob:1.0})
    gg = sess.run(model.gg, feed_dict={model.x:x, model.y:y, model.keep_prob:2.0, model.v:v})
    from IPython import embed; embed()

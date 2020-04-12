import os
import numpy as np
import torch

class CIFAR10(object):
    def __init__(self, data_dir):
        super(CIFAR10, self).__init__()
        self.n_classes = 10

        train = np.load(os.path.join(data_dir, 'train.npz'))
        test = np.load(os.path.join(data_dir, 'test.npz'))
        self.X_train = train['image']
        self.Y_train = train['label']
        self.X_test = test['image']
        self.Y_test = test['label']

        self.n_test = len(self.Y_test)
        self.n_train = len(self.Y_train)

        self.trans()

    def __str__(self):
        return 'CIFAR10\nnum_train: %d\nnum_test: %d' % (self.n_train, self.n_test)

    def transpose(self):
        self.X_train = np.transpose(self.X_train, [0,3,1,2])
        self.X_test = np.transpose(self.X_test, [0,3,1,2])

    def normalize(self):
        self.X_train = self.X_train / 255.0
        self.X_test = self.X_test / 255.0

    def trans(self):
        self.transpose()
        self.normalize()

    def getTrainBatch(self, batch_size, cuda=False):
        mask = np.random.choice(self.n_train, batch_size, False)
        X = torch.FloatTensor(self.X_train[mask])
        Y = torch.LongTensor(self.Y_train[mask])
        if cuda:
            X = X.cuda()
            Y = Y.cuda()
        return X, Y

    def getTrainBatchList(self, batch_size, n_batch=100, cuda=False):
        batch_list = []
        for i in range(n_batch):
            X, Y = self.getTrainBatch(batch_size, cuda)
            batch_list.append((X, Y))
        return batch_list

    def getTestList(self, batch_size=500, cuda=False):
        n_batch = self.n_test // batch_size
        batch_list = []
        for i in range(n_batch):
            X = torch.FloatTensor(self.X_test[batch_size*i:batch_size*(i+1)])
            Y = torch.LongTensor(self.Y_test[batch_size*i:batch_size*(i+1)])
            if cuda:
                X = X.cuda()
                Y = Y.cuda()
            batch_list.append((X, Y))
        return batch_list

    def getTrainList(self, batch_size=5000, cuda=False):
        n_batch = self.n_train // batch_size
        batch_list = []
        for i in range(n_batch):
            X = torch.FloatTensor(self.X_train[batch_size*i:batch_size*(i+1)])
            Y = torch.LongTensor(self.Y_train[batch_size*i:batch_size*(i+1)])
            if cuda:
                X = X.cuda()
                Y = Y.cuda()
            batch_list.append((X, Y))
        return batch_list

if __name__ == '__main__':
    datapath = '/home/wjf/datasets/CIFAR10/numpy'
    dataset = CIFAR10(datapath)
    from IPython import embed; embed()

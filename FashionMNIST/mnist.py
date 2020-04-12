import os
import numpy as np

class MNIST(object):
    def __init__(self, data_file):
        super(MNIST, self).__init__()
        self.data = np.load(data_file)
        self.X = self.data['image']
        self.Y = self.data['label']
        assert len(self.X) == len(self.Y)
        self.n_examples = len(self.X)
        self.n_classes = 10

        self.X, self.Y = self.trans(self.X, self.Y)

    def onehot(self, labels):
        return np.eye(self.n_classes)[labels]

    def normalize(self, images):
        images = (images - 127.5) / 127.5
        return images

    def trans(self, images, labels):
        labels = self.onehot(labels)
        images = self.normalize(images)
        return images, labels

if __name__ == '__main__':
    HOME = os.environ['HOME']
    DATASET = os.path.join(HOME, 'datasets/fashion')
    train = MNIST(os.path.join(DATASET, 'train.npz'))
    val = MNIST(os.path.join(DATASET, 'val.npz'))
    from IPython import embed; embed()

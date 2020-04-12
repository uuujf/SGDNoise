import numpy as np
import scipy.io as sio
import os

def save2npz(images, labels, out_file):
    assert len(images) == len(labels)
    np.savez(out_file, image=images, label=labels)
    print('Save data to %s'%(out_file))
    return True

if __name__ == '__main__':
    height, width, channel = 32, 32, 3
    HOME = os.environ['HOME']
    DATASET = os.path.join(HOME, 'datasets/SVHN')
    TARGET = os.path.join(DATASET, 'train25000_test70000')

    trainset = sio.loadmat(os.path.join(DATASET, 'raw/train_32x32.mat'))
    images = trainset['X'].transpose([3,0,1,2])
    labels = trainset['y'].reshape(-1)
    labels = labels - 1
    save2npz(images[0:70000], labels[0:70000], os.path.join(TARGET, 'test.npz'))
    testset = sio.loadmat(os.path.join(DATASET, 'raw/test_32x32.mat'))
    images = testset['X'].transpose([3,0,1,2])
    labels = testset['y'].reshape(-1)
    labels = labels - 1
    save2npz(images[0:25000], labels[0:25000], os.path.join(TARGET, 'train.npz'))

    # test
    train = np.load(os.path.join(TARGET, 'train.npz'))
    test = np.load(os.path.join(TARGET, 'test.npz'))
    from IPython import embed; embed()

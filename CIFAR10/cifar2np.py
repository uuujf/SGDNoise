import argparse
import gzip
import os
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save2npz(images, labels, out_file):
    assert len(images) == len(labels)
    np.savez(out_file, image=images, label=labels)
    print('Save data to %s'%(out_file))
    return True

if __name__ == '__main__':
    n_examples = 10000
    height, width, channel = 32, 32, 3
    HOME = os.environ['HOME']
    DATASET = os.path.join(HOME, 'datasets/CIFAR10')
    TARGET = os.path.join(DATASET, 'numpy')

    # convet train data
    print('read train files')
    all_images = []
    all_labels = []
    for i in range(5):
        train_raw = unpickle(os.path.join(DATASET, 'cifar-10-batches-py/data_batch_'+str(i+1)))
        images = train_raw[b'data'].reshape(n_examples, channel, height, width)
        images = np.swapaxes(images, 1, 2)
        images = np.swapaxes(images, 2, 3)
        labels = np.array(train_raw[b'labels'], dtype=np.uint8)
        all_images.append(images)
        all_labels.append(labels)
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print('convert train files')
    save2npz(all_images, all_labels, os.path.join(TARGET, 'train.npz'))

    # convet validation data
    print('read val files')
    val_raw = unpickle(os.path.join(DATASET, 'cifar-10-batches-py/test_batch'))
    images = val_raw[b'data'].reshape(n_examples, channel, height, width)
    images = np.swapaxes(images, 1, 2)
    images = np.swapaxes(images, 2, 3)
    labels = np.array(val_raw[b'labels'], dtype=np.uint8)
    print('convert val files')
    save2npz(images, labels, os.path.join(TARGET, 'test.npz'))

    # test
    train = np.load(os.path.join(TARGET, 'train.npz'))
    test = np.load(os.path.join(TARGET, 'test.npz'))
    from IPython import embed; embed()

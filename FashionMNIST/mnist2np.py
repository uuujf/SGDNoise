import argparse
import gzip
import os
import numpy as np

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(f):
    """Extract the images into a 4D uint8 np array [index, y, x, depth].
    Args:
        f: A file object that can be passed into a gzip reader.
    Returns:
        data: A 4D uint8 np array [index, y, x, depth].
    Raises:
        ValueError: If the bytestream does not start with 2051.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %(magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 np array [index].
    Args:
        f: A file object that can be passed into a gzip reader.
        one_hot: Does one hot encoding for the result.
        num_classes: Number of classes for the one hot encoding.
    Returns:
        labels: a 1D uint8 np array.
    Raises:
        ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                        (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return dense_to_one_hot(labels, num_classes)
    return labels

def save2npz(images, labels, out_file, marks=None):
    assert len(images) == len(labels)
    if marks is None:
        marks = np.zeros_like(labels)
    np.savez(out_file, image=images, label=labels, mark=marks)
    print('Save data to %s'%(out_file))
    return True

def main():
    #height, width, channel = 28, 28, 1
    HOME = os.environ['HOME']
    DATASET = os.path.join(HOME, 'datasets/fashion_tf_1k_cor')

    # convet train data
    print('read train files')
    with open(os.path.join(DATASET, 'raw/train-images-idx3-ubyte.gz'), 'rb') as f:
        images = extract_images(f)[0:6000]
    with open(os.path.join(DATASET, 'raw/train-labels-idx1-ubyte.gz'), 'rb') as f:
        labels = extract_labels(f)[0:6000]
    print('convert corrupted train files')
    rnd_labels = np.random.choice(10, 200)
    labels_fake = labels.copy()
    labels_fake[1000:1200] = rnd_labels
    marks = (labels != labels_fake).astype(np.int8)
    save2npz(images, labels_fake, os.path.join(DATASET, 'train.npz'), marks)

    # convet validation data
    print('read val files')
    with open(os.path.join(DATASET, 'raw/t10k-images-idx3-ubyte.gz'), 'rb') as f:
        images = extract_images(f)
    with open(os.path.join(DATASET, 'raw/t10k-labels-idx1-ubyte.gz'), 'rb') as f:
        labels = extract_labels(f)
    print('convert val files')
    save2npz(images, labels, os.path.join(DATASET, 'val.npz'))

if __name__ == '__main__':
    main()
    HOME = os.environ['HOME']
    DATASET = os.path.join(HOME, 'datasets/fashion_tf_1k_cor')
    train = np.load(os.path.join(DATASET, 'train.npz'))
    val = np.load(os.path.join(DATASET, 'val.npz'))
    from IPython import embed; embed()

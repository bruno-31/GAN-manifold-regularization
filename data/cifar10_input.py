# CIFAR10 Downloader

import pickle
import os
import errno
import tarfile
import shutil
import numpy as np

import urllib3


_shuffle = True


def _unpickle_file(filename):
    # print("Loading pickle file: {}".format(filename))

    with open(filename, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    # Reorder the data
    img = data[b'data']
    img = img.reshape([-1, 3, 32, 32])
    img = img.transpose([0, 2, 3, 1])
    # Load labels
    lbl = np.array(data[b'labels'])

    return img, lbl


def _get_dataset(path,split):
    assert split == "test" or split == "train"
    dirname = "cifar-10-batches-py"
    # data_url = "http://10.217.128.198/datasets/cifar-10-python.tar.gz"
    # data_url = "http://10.217.128.198/mnt/data_c/datasets/cifar-10-python.tar.gz"
    data_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    if not os.path.exists(os.path.join(path, dirname)):
        # Extract or download data
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        file_path = os.path.join(path, data_url.split('/')[-1])
        if not os.path.exists(file_path):
            # Download
            print("Downloading {}".format(data_url))
            with urllib3.PoolManager().request('GET', data_url, preload_content=False) as r, \
                    open(file_path, 'wb') as w:
                shutil.copyfileobj(r, w)

        print("Unpacking {}".format(file_path))
        # Unpack data
        tarfile.open(name=file_path, mode="r:gz").extractall(path)

    # Import the data
    filenames = ["test_batch"] if split == "test" else \
        ["data_batch_{}".format(i) for i in range(1, 6)]

    imgs = []
    lbls = []
    for f in filenames:
        img, lbl = _unpickle_file(os.path.join(path, dirname, f))
        imgs.append(img)
        lbls.append(lbl)

    # Now we flatten the arrays
    imgs = np.concatenate(imgs)
    lbls = np.concatenate(lbls)

    # Convert images to [0..1] range
    imgs = imgs.astype(np.float32) / 255.0
    imgs = imgs *2. -1.
    # Convert images to [-1..1] range
    # imgs = (imgs.astype(np.float32)-127.5) / 128.
    # Convert label to one hot encoding
    # lbl = np.zeros((len(lbls),10)) #lbl s !!
    # lbl[np.arange(len(lbls)), lbls] = 1
    return imgs, lbls.astype(np.uint8)


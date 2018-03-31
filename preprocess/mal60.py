import numpy as np

FNAME = './mal60.npz'


def load(path):
    with np.load(path) as data:
        train = data['data']
        train_labels = data['labels']

    return train, train_labels


train, train_labels = load(FNAME)
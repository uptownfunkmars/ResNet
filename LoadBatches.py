import numpy as np
import cv2
import glob
import itertools
import matplotlib.pyplot as plt
import random

#相对路径这样写也可以
file = "./CIFAR-10/cifar-10-batches-py/data_batch_1"

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def dataset(batch_size, input_height, input_width, dict):
    img = dict[b'data']
    lab = dict[b'labels']

    zipped = itertools.cycle(zip(img, lab))

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            images, labels = zipped.__next__()
            im = images.reshaped((input_height, input_width, 3))

            X.append(im)
            Y.append(labels)

        yield np.array(X), np.array(Y)
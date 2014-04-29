__author__ = 'adeb'

import cPickle
import gzip
import os
import sys
import time

import numpy

import h5py

import theano
import theano.tensor as T

dataset = 'mnist.pkl.gz'

data_dir, data_file = os.path.split(dataset)
if data_dir == "" and not os.path.isfile(dataset):
    # Check if dataset is in the data directory.
    new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
    if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
        dataset = new_path

print '... loading data'

# Load the dataset
f = gzip.open(dataset, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

n_classes = 10

train_x = numpy.concatenate((train_set[0], valid_set[0]))
train_y0 = numpy.concatenate((train_set[1], valid_set[1]))
train_y = numpy.zeros((train_x.shape[0], n_classes))
train_y[range(train_x.shape[0]), train_y0] = 1

test_x = test_set[0]
test_y0 = test_set[1]
test_y = numpy.zeros((test_x.shape[0], n_classes))
test_y[range(test_x.shape[0]), test_y0] = 1


f = h5py.File('../data/training_mnist.h5', 'w')
f['/inputs'] = train_x.transpose()
f['/targets'] = train_y.transpose()
f.attrs['n_classes'] = n_classes
f.attrs['patch_width'] = 28
f.attrs['n_samples'] = train_x.shape[0]
f.close()


f = h5py.File('../data/testing_mnist.h5', 'w')
f['/inputs'] = test_x.transpose()
f['/targets'] = test_y.transpose()
f.attrs['n_patch_per_voxel'] = 1
f.close()


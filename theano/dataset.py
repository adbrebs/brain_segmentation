__author__ = 'adeb'

import numpy

import h5py

import theano


class Dataset():
    def __init__(self, training_data, testing_data):

        print '... loading data'

        # Load training data
        f = h5py.File(training_data, driver='core', backing_store=False)
        set_x = f['/inputs'].value.transpose()
        set_y = f['/targets'].value.transpose()
        self.n_classes = int(f.attrs['n_classes'])
        self.patch_width = int(f.attrs['patch_width'])
        self.n_data = int(f.attrs['n_samples'])
        f.close()

        # Create a validation set
        validatioin_split = int(0.9 * self.n_data)
        train_x = set_x[0:validatioin_split-1, :]
        train_y = set_y[0:validatioin_split-1, :]
        valid_x = set_x[validatioin_split:self.n_data, :]
        valid_y = set_y[validatioin_split:self.n_data, :]

        # Load testing data
        f = h5py.File(testing_data, driver='core', backing_store=False)
        test_x = f['/inputs'].value.transpose()
        test_y = f['/targets'].value.transpose()
        self.n_patch_per_voxel_testing = int(f.attrs['n_patch_per_voxel'])
        f.close()

        # Store the data in shared variables
        def share_data(data, borrow=True):
            shared_data = theano.shared(numpy.asarray(data, dtype=theano.config.floatX), borrow=borrow)
            return shared_data
        self.test_x = share_data(test_x)
        self.test_y = share_data(test_y)
        self.valid_x = share_data(valid_x)
        self.valid_y = share_data(valid_y)
        self.train_x = share_data(train_x)
        self.train_y = share_data(train_y)

        self.n_train = self.train_x.get_value(borrow=True).shape[0]
        self.n_valid = self.valid_x.get_value(borrow=True).shape[0]
        self.n_test = self.test_x.get_value(borrow=True).shape[0]


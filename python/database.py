__author__ = 'adeb'

import numpy

import theano

from data_generation import Dataset

class DataBase():
    def __init__(self, config):

        training_data_file = config.get('dataset', 'training_data')
        testing_data_file = config.get('dataset', 'testing_data')

        print '... loading data ' + training_data_file + ' and ' + testing_data_file

        # Load training data
        training_data = Dataset()
        training_data.read(training_data_file)
        self.n_classes = training_data.n_classes
        self.patch_width = training_data.patch_width
        n_data = training_data.n_patches

        # Create a validation set
        validatioin_split = int(0.9 * n_data)
        train_x = training_data.patch[0:validatioin_split-1, :]
        train_y = training_data.tg[0:validatioin_split-1, :]
        valid_x = training_data.patch[validatioin_split:n_data, :]
        valid_y = training_data.tg[validatioin_split:n_data, :]

        # Load testing data
        testing_data = Dataset()
        testing_data.read(testing_data_file)
        self.n_patch_per_voxel_testing = testing_data.n_patch_per_voxel
        test_x = testing_data.patch
        test_y = testing_data.tg

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
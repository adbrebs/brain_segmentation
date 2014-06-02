__author__ = 'adeb'

import numpy as np

import theano

from utilities import share
from dataset import DatasetBrainParcellation


class DataBase():
    """
    Abstract class responsible for splitting datasets into training, validating and testing datasets. It also loads them on
    the GPU.

    Attributes:
        test_in, test_out, valid_in, valid_out, train_in, train_out (Theano shared 2D matrices): data sets loaded on
            the GPU.
    """
    def __init__(self):

        self.n_in_features = None
        self.n_out_features = None
        self.n_data = None

        self.test_in = None
        self.test_out = None
        self.valid_in = None
        self.valid_out = None
        self.train_in = None
        self.train_out = None

        self.n_train = None
        self.n_valid = None
        self.n_test = None

    def init_from_config(self, config):
        raise NotImplementedError

    def share_data(self, test_in, test_out, valid_in, valid_out, train_in, train_out):
        """
        Store the data in shared variables
        """
        self.test_in = share(test_in)
        self.test_out = share(test_out)
        self.valid_in = share(valid_in)
        self.valid_out = share(valid_out)
        self.train_in = share(train_in)
        self.train_out = share(train_out)

        self.n_train, self.n_in_features = train_in.shape
        self.n_valid = valid_in.shape[0]
        self.n_test = test_in.shape[0]

        self.n_out_features = train_out.shape[1]
        self.n_data = self.n_train + self.n_valid + self.n_test


class DataBaseBrainParcellation(DataBase):
    """
    Attributes:
        patch_width (int): width of an input patch
        n_patch_per_voxel_testing (int): number of patches per unique voxel in the testing data set. This allows the
            output of a voxel to be predicted from several patches.
    """
    def __init__(self):
        DataBase.__init__(self)

        self.patch_width = None
        self.n_patch_per_voxel_testing = None

    def init_from_config(self, config):

        training_data_file = config.training_data_path
        testing_data_file = config.testing_data_path

        print '... loading data ' + training_data_file + ' and ' + testing_data_file

        # Load training and testing data
        training_data = DatasetBrainParcellation()
        training_data.read(training_data_file)
        testing_data = DatasetBrainParcellation()
        testing_data.read(testing_data_file)
        if training_data.n_in_features != testing_data.n_in_features:
            raise Exception("The training and testing datasets do not have the same number of input features")
        if training_data.n_out_features != testing_data.n_out_features:
            raise Exception("The training and testing datasets do not have the same number of outputs")

        self.n_out_features = training_data.n_out_features
        self.patch_width = training_data.patch_width
        self.n_patch_per_voxel_testing = testing_data.n_patch_per_voxel
        n_data = training_data.n_data

        # Create a validation set
        prop_validation = config.prop_validation
        validatioin_split = int((1-prop_validation) * n_data)
        train_x = training_data.inputs[0:validatioin_split, :]
        train_y = training_data.outputs[0:validatioin_split, :]
        valid_x = training_data.inputs[validatioin_split:n_data, :]
        valid_y = training_data.outputs[validatioin_split:n_data, :]

        # Testing data
        test_x = testing_data.inputs
        test_y = testing_data.outputs

        # Transform the data into Theano shared variables
        self.share_data(test_x, test_y, valid_x, valid_y, train_x, train_y)
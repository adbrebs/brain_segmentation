__author__ = 'adeb'

import ConfigParser

import theano.sandbox.cuda

from convolutional_mlp import *


if __name__ == '__main__':
    config = ConfigParser.ConfigParser()
    config.read(str(sys.argv[1]))
    training_data = config.get('general', 'training_data')
    testing_data = config.get('general', 'testing_data')
    theano.sandbox.cuda.use((config.get('general', 'gpu')))
    evaluate_lenet5(training_data, testing_data)
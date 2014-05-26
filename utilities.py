__author__ = 'adeb'

import numpy as np

import theano


def share(value, name=None, borrow=True):
    return theano.shared(np.asarray(value, dtype=theano.config.floatX), name=name, borrow=borrow)
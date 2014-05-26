__author__ = 'adeb'

import numpy as np

import theano
import theano.tensor as T

from utilities import share


class LearningUpdate():
    def __init__(self):
        pass

    def compute_updates(self, params, grads):
        raise NotImplementedError


class LearningUpdateGD(LearningUpdate):
    def __init__(self, learning_rate):
        LearningUpdate.__init__(self)
        self.learning_rate = share(learning_rate, "learning_rate")

    def compute_updates(self, params, grads):
        updates = []
        for param_i, grad_i in zip(params, grads):
            updates.append((param_i, param_i - self.learning_rate * grad_i))
        return updates


class LearningUpdateGDMomentum(LearningUpdate):
    def __init__(self, learning_rate, momentum):
        LearningUpdate.__init__(self)
        self.learning_rate = share(learning_rate, "learning_rate")
        self.momentum = share(momentum, "momentum")
        if momentum < 0 or momentum > 1:
            raise Exception("Momentum value should be between 0 and 1.")

    def compute_updates(self, params, grads):
        updates = []
        for param_i, grad_i in zip(params, grads):
            diff = share(np.zeros(param_i.get_value().shape, dtype=theano.config.floatX), "diff")
            update_diff = self.momentum * diff - self.learning_rate * grad_i
            updates.append((param_i, param_i + update_diff))
            updates.append((diff, update_diff))
        return updates
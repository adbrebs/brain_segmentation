__author__ = 'adeb'

from theano import tensor as T


class NeuronType():
    def __init__(self):
        pass

    def activation_function(self, x):
        raise NotImplementedError


class NeuronTanh(NeuronType):
    def __init__(self):
        NeuronType.__init__(self)

    def activation_function(self, x):
        return T.tanh(x)


class NeuronSoftmax(NeuronType):
    def __init__(self):
        NeuronType.__init__(self)

    def activation_function(self, x):
        return T.nnet.softmax(x)


class NeuronRELU(NeuronType):
    def __init__(self):
        NeuronType.__init__(self)

    def activation_function(self, x):
        return T.maximum(0, x)

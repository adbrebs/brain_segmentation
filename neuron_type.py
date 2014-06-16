__author__ = 'adeb'

from theano import tensor as T


class NeuronType():
    """
    Abstract class defining a neuron type. This class defines the activation function of the neuron.
    """
    def __init__(self):
        self.name = None

    def activation_function(self, x):
        raise NotImplementedError

    def __str__(self):
        return "Neuron type: {}".format(self.name)


class NeuronTanh(NeuronType):
    def __init__(self):
        NeuronType.__init__(self)
        self.name = "Tanh"

    def activation_function(self, x):
        return T.tanh(x)


class NeuronSoftmax(NeuronType):
    def __init__(self):
        NeuronType.__init__(self)
        self.name = "SoftMax"

    def activation_function(self, x):
        return T.nnet.softmax(x)


class NeuronRELU(NeuronType):
    """
    Rectified linear unit
    """
    def __init__(self):
        NeuronType.__init__(self)
        self.name = "RELU"

    def activation_function(self, x):
        return T.switch(x > 0., x, 0)

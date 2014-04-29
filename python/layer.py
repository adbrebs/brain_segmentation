__author__ = 'adeb'


import numpy

import theano
import theano.tensor as T


class Layer():

    def __init__(self, n_in, n_out, x):
        """
        :type x: theano.tensor.TensorType
        :param x: symbolic variable that describes the input of the layer

        :type n_in: int
        :param n_in: number of input units

        :type n_out: int
        :param n_out: number of output units
        """
        self.n_in = n_in
        self.n_out = n_out
        self.x = x

    def mse(self, y_true):
        """Return the mean square error.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  corresponding output
        """
        return T.mean(T.sum((self.y - y_true) * (self.y - y_true), axis=1))

    def errors(self, y_true):
        return T.mean(T.neq(T.argmax(self.y, axis=1), T.argmax(y_true, axis=1)))

    def negative_log_likelihood(self, y_true):
        return -T.mean(T.sum(T.log(self.y) * y_true, axis=1))


class LayerFullyConnected(Layer):
    def __init__(self, n_in, n_out, x):
        Layer.__init__(self, n_in, n_out, x)

        self.w, self.b = self.init()

        self.params = [self.w, self.b]

    def init(self):
        raise NotImplementedError


class LayerTan(LayerFullyConnected):
    def __init__(self, n_in, n_out, x):
        LayerFullyConnected.__init__(self, n_in, n_out, x)

        self.y = T.tanh(T.dot(x, self.w) + self.b)

    def init(self):
        w_values = numpy.asarray(numpy.random.uniform(
            low=-numpy.sqrt(6. / (self.n_in + self.n_out)),
            high=numpy.sqrt(6. / (self.n_in + self.n_out)),
            size=(self.n_in, self.n_out)), dtype=theano.config.floatX)

        b_values = numpy.zeros((self.n_out,), dtype=theano.config.floatX)

        w = theano.shared(w_values, name='w', borrow=True)
        b = theano.shared(b_values, name='b', borrow=True)

        return w, b

class LayerSoftMax(LayerFullyConnected):
    def __init__(self, n_in, n_out, x):
        LayerFullyConnected.__init__(self, n_in, n_out, x)

        self.y = T.nnet.softmax(T.dot(x, self.w) + self.b)

    def init(self):
        w = theano.shared(value=numpy.zeros((self.n_in, self.n_out),
                                                 dtype=theano.config.floatX),
                               name='w', borrow=True)

        b = theano.shared(value=numpy.zeros((self.n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        return w, b


class LayerConv2D(Layer):
    def __init__(self, n_in, n_out, x):
        Layer.__init__(self, n_in, n_out, x)

        self.w, self.b = self.init()

        self.params = [self.w, self.b]
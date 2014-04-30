__author__ = 'adeb'


import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv


class Layer():

    def __init__(self, x):
        """
        :type x: theano.tensor.TensorType
        :param x: symbolic variable that describes the input of the layer
        """
        self.x = x

        self.y = 0

    def mse(self, y_true):
        """Return the mean square error.

        :type y_true: theano.tensor.TensorType
        :param y_true: corresponds to a vector that gives for each example the
                  corresponding output
        """
        return T.mean(T.sum((self.y - y_true) * (self.y - y_true), axis=1))

    def errors(self, y_true):
        return T.mean(T.neq(T.argmax(self.y, axis=1), T.argmax(y_true, axis=1)))

    def negative_log_likelihood(self, y_true):
        return -T.mean(T.sum(T.log(self.y) * y_true, axis=1))


class LayerFullyConnected(Layer):
    def __init__(self, n_in, n_out, x):
        Layer.__init__(self, x)

        self.n_in = n_in
        self.n_out = n_out

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
        w_values = numpy.zeros((self.n_in, self.n_out), dtype=theano.config.floatX)
        w = theano.shared(w_values, name='w', borrow=True)

        b_values = numpy.zeros((self.n_out,), dtype=theano.config.floatX)
        b = theano.shared(b_values, name='b', borrow=True)

        return w, b


class LayerConvPool2D(Layer):
    def __init__(self, x, image_shape, filter_shape, poolsize=(2, 2)):
        """
        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
        Layer.__init__(self, x)

        assert image_shape[1] == filter_shape[1]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])

        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        w_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.w = theano.shared(numpy.asarray(
            numpy.random.uniform(low=-w_bound, high=w_bound, size=filter_shape),
            dtype=theano.config.floatX), borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=x,
                               filters=self.w,
                               filter_shape=filter_shape,
                               image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize,
                                            ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.y = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.w, self.b]
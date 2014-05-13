__author__ = 'adeb'


import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import neurons


class Layer():
    def __init__(self, neuron_type):
        self.w = None
        self.b = None
        self.neuron_type = neuron_type

    def forward(self, x, batch_size):
        """Return the output of the layer
        Args:
            x (theano.tensor.TensorType): input of the layer
        Returns:
            (theano.tensor.TensorType): output of the layer
        """
        raise NotImplementedError

    def save_parameters(self, h5file, name):
        h5file.create_dataset(name + "/w", data=self.w.get_value(), dtype='f')
        h5file.create_dataset(name + "/b", data=self.b.get_value(), dtype='f')

    def load_parameters(self, h5file, name):
        self.w.set_value(h5file[name + "/w"].value)
        self.b.set_value(h5file[name + "/b"].value)


class LayerFullyConnected(Layer):
    """
    Layer in which each input is connected to all the layer neurones
    """
    def __init__(self, neuron_type, n_in, n_out):
        Layer.__init__(self, neuron_type)

        self.n_in = n_in
        self.n_out = n_out

        self.w, self.b = self.init()

        self.params = [self.w, self.b]

    def init(self):
        """
        Initialize the parameters of the layer
        """
        w_values = numpy.asarray(numpy.random.uniform(
            low=-numpy.sqrt(6. / (self.n_in + self.n_out)),
            high=numpy.sqrt(6. / (self.n_in + self.n_out)),
            size=(self.n_in, self.n_out)), dtype=theano.config.floatX)

        b_values = 0.1 + numpy.zeros((self.n_out,), dtype=theano.config.floatX)

        w = theano.shared(w_values, name='w', borrow=True)
        b = theano.shared(b_values, name='b', borrow=True)

        return w, b

    def forward(self, x, batch_size):
        return self.neuron_type.activation_function(T.dot(x, self.w) + self.b)


class LayerConv2D(Layer):
    """
    Convolution + pooling layer
    """
    def __init__(self, neuron_type, image_shape, filter_shape, poolsize=(2, 2)):
        """
        Args:
            image_shape (tuple or list of length 3):
            (num input feature maps, image height, image width)

            filter_shape (tuple or list of length 4):
            (number of filters, num input feature maps, filter height,filter width)
        """
        Layer.__init__(self, neuron_type)
        self.image_shape = image_shape
        self.filter_shape = filter_shape

        assert image_shape[0] == filter_shape[1]

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

        self.params = [self.w, self.b]

    def forward(self, x, batch_size):
        img_batch_shape = (batch_size,) + self.image_shape

        x = x.reshape(img_batch_shape)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=x,
                               filters=self.w,
                               image_shape=img_batch_shape,
                               filter_shape=self.filter_shape)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        return self.neuron_type.activation_function(conv_out + self.b.dimshuffle('x', 0, 'x', 'x')).flatten(2)


class LayerConvPool2D(Layer):
    """
    Convolution + pooling layer
    """
    def __init__(self, neuron_type, image_shape, filter_shape, poolsize=(2, 2)):
        """
        Args:
            image_shape (tuple or list of length 3):
            (num input feature maps, image height, image width)

            filter_shape (tuple or list of length 4):
            (number of filters, num input feature maps, filter height,filter width)

            poolsize (tuple or list of length 2):
            the downsampling (pooling) factor (#rows,#cols)
        """
        Layer.__init__(self, neuron_type)
        self.image_shape = image_shape
        self.filter_shape = filter_shape
        self.poolsize = poolsize

        assert image_shape[0] == filter_shape[1]

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
        b_values = 0.1 + numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        self.params = [self.w, self.b]

    def forward(self, x, batch_size):
        img_batch_shape = (batch_size,) + self.image_shape

        x = x.reshape(img_batch_shape)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=x,
                               filters=self.w,
                               image_shape=img_batch_shape,
                               filter_shape=self.filter_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=self.poolsize,
                                            ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        return self.neuron_type.activation_function(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')).flatten(2)
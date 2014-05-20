__author__ = 'adeb'


import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv, conv3d2d

from max_pool_3d import max_pool_3d


class Layer():
    """
    Abstract class defining a layer of neurons.
    """
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


class LayerConv2DAbstract(Layer):
    """
    Abstract class defining common components of LayerConv2D and LayerConvPool2D
    """
    def __init__(self, neuron_type, image_shape, filter_shape):
        """
        Args:
            image_shape (tuple or list of length 3):
            (num input feature maps, image height, image width)

            filter_shape (tuple or list of length 4):
            (number of filters, num input feature maps, filter height, filter width)
        """
        Layer.__init__(self, neuron_type)
        self.image_shape = image_shape
        self.filter_shape = filter_shape

        assert image_shape[0] == filter_shape[1]

        fan_in, fan_out = self.init_bounds_parameters()

        # initialize weights with random weights
        w_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.w = theano.shared(numpy.asarray(
            numpy.random.uniform(low=-w_bound, high=w_bound, size=filter_shape),
            dtype=theano.config.floatX), borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        self.params = [self.w, self.b]

    def init_bounds_parameters(self):
        raise NotImplementedError

    def forward(self, x, batch_size):
        img_batch_shape = (batch_size,) + self.image_shape

        x = x.reshape(img_batch_shape)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=x,
                               filters=self.w,
                               image_shape=img_batch_shape,
                               filter_shape=self.filter_shape)

        return self.forward_virtual(conv_out)

    def forward_virtual(self, conv_out):
        raise NotImplementedError


class LayerConv2D(LayerConv2DAbstract):
    """
    2D Convolution layer
    """
    def __init__(self, neuron_type, image_shape, filter_shape):
        LayerConv2DAbstract.__init__(self, neuron_type, image_shape, filter_shape)

    def init_bounds_parameters(self):
        fan_in = numpy.prod(self.filter_shape[1:])
        fan_out = self.filter_shape[0] * numpy.prod(self.filter_shape[2:])
        return fan_in, fan_out

    def forward_virtual(self, conv_out):
        return self.neuron_type.activation_function(conv_out + self.b.dimshuffle('x', 0, 'x', 'x')).flatten(2)


class LayerConvPool2D(LayerConv2DAbstract):
    """
    Convolution + pooling layer. The reason for not having a separate pooling layer is that the combination of the two
    layers can be optimized.
    """
    def __init__(self, neuron_type, image_shape, filter_shape, poolsize=(2, 2)):
        self.poolsize = poolsize
        LayerConv2DAbstract.__init__(self, neuron_type, image_shape, filter_shape)

    def init_bounds_parameters(self):
        fan_in = numpy.prod(self.filter_shape[1:])
        fan_out = (self.filter_shape[0] * numpy.prod(self.filter_shape[2:]) / numpy.prod(self.poolsize))
        return fan_in, fan_out

    def forward_virtual(self, conv_out):
        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=self.poolsize,
                                            ignore_border=True)

        return self.neuron_type.activation_function(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')).flatten(2)


class LayerConvPool3D(Layer):
    """
    Convolution + pooling layer
    """
    def __init__(self, neuron_type, in_channels, in_shape,
                 flt_channels, flt_shape, poolsize):
        """
        Args:
            image_shape (tuple or list of length 4):
            (image depth, num input feature maps, image height, image width)

            filter_shape (tuple or list of length 5):
            (number of filters, filter depth, num input feature maps, filter height,filter width)
        """
        Layer.__init__(self, neuron_type)

        in_width, in_height, in_depth = in_shape
        flt_depth, flt_height, flt_width = flt_shape

        self.image_shape = image_shape = (in_depth, in_channels, in_height, in_width)
        self.filter_shape = filter_shape = (flt_channels, flt_depth, in_channels, flt_height, flt_width)
        self.poolsize = poolsize

        fan_in = in_width * in_height * in_depth
        fan_out = flt_channels * flt_width * flt_height * flt_depth

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
        conv_out = conv3d2d.conv3d(signals=x,
                                   filters=self.w,
                                   signals_shape=img_batch_shape,
                                   filters_shape=self.filter_shape,
                                   border_mode='valid')

        pooled_out = max_pool_3d(conv_out.dimshuffle([0,2,1,3,4]), self.poolsize, ignore_border=True)

        return self.neuron_type.activation_function(pooled_out.dimshuffle([0,2,1,3,4]) + self.b.dimshuffle('x', 'x', 0, 'x', 'x')).flatten(2)

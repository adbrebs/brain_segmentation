__author__ = 'adeb'


import numpy as np

import theano
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv, conv3d2d

from utilities import share, get_h5file_data
from max_pool_3d import max_pool_3d


class LayerBlock():
    """
    Abstract class that represents a function. It is the building block of a layer
    """
    def __init__(self):
        self.name = None
        self.params = None

    def forward(self, x, batch_size):
        """Return the output of the layer block
        Args:
            x (theano.tensor.TensorType): input of the layer block
        Returns:
            (theano.tensor.TensorType): output of the layer block
        """
        raise NotImplementedError

    def save_parameters(self, h5file, name):
        """
        Save all parameters of the layer block in a hdf5 file.
        """
        pass

    def load_parameters(self, h5file, name):
        """
        Load all parameters of the layer block in a hdf5 file.
        """
        pass

    def __str__(self):
        raise NotImplementedError


class LayerBlockIdentity(LayerBlock):
    """
    Identity function
    """
    def __init__(self):
        LayerBlock.__init__(self)
        self.name = "Identity Layer block"

    def forward(self, x, batch_size):
        return x

    def __str__(self):
        msg = "[{}]] \n".format(self.name)
        return msg


class LayerBlockOfNeurons(LayerBlock):
    """
    Abstract class defining a group of neurons.

    Attributes:
        name (string): Name of the layer block (used for printing or writing)
        w (theano shared numpy array): Weights of the layer block
        b (theano shared numpy array): Biases of the layer block
        params (list): [w,b]
        neuron_type (NeuronType object): defines the type of the neurons of the layer block
    """
    def __init__(self, neuron_type):
        LayerBlock.__init__(self)
        self.w = None
        self.b = None
        self.neuron_type = neuron_type

    def init_parameters(self, w_shape, b_shape):
        w_bound = self.compute_bound_parameters_virtual()

        # initialize weights with random weights
        self.w = share(np.asarray(
            np.random.uniform(low=-w_bound, high=w_bound, size=w_shape),
            dtype=theano.config.floatX), "w")

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = 0.1 + np.zeros(b_shape, dtype=theano.config.floatX)  # Slightly positive for RELU units
        self.b = share(b_values, "b")

        self.params = [self.w, self.b]

    def compute_bound_parameters_virtual(self):
        raise NotImplementedError

    def save_parameters(self, h5file, name):
        h5file.create_dataset(name + "/w", data=self.w.get_value(), dtype='f')
        h5file.create_dataset(name + "/b", data=self.b.get_value(), dtype='f')

    def load_parameters(self, h5file, name):
        self.w.set_value(get_h5file_data(h5file, name + "/w"))
        self.b.set_value(get_h5file_data(h5file, name + "/b"))

    def __str__(self):
        msg = "[{}] with [{}] \n".format(self.name, self.neuron_type)
        msg += self.print_virtual()
        n_parameters = 0
        for p in self.params:
            n_parameters += p.get_value().size
        msg += "Number of parameters: {} \n".format(n_parameters)
        return msg

    def print_virtual(self):
        return ""


class LayerBlockFullyConnected(LayerBlockOfNeurons):
    """
    Layer block in which each input is connected to all the layer neurons
    """
    def __init__(self, neuron_type, n_in, n_out):
        LayerBlockOfNeurons.__init__(self, neuron_type)

        self.name = "Fully connected layer block"
        self.n_in = n_in
        self.n_out = n_out

        self.init_parameters((self.n_in, self.n_out), (self.n_out,))

    def compute_bound_parameters_virtual(self):
        return np.sqrt(6. / (self.n_in + self.n_out))

    def forward(self, x, batch_size):
        return self.neuron_type.activation_function(theano.tensor.dot(x, self.w) + self.b)

    def print_virtual(self):
        return "Number of inputs: {} \nNumber of outputs: {}\n".format(self.n_in, self.n_out)


class LayerBlockConv2DAbstract(LayerBlockOfNeurons):
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
        LayerBlockOfNeurons.__init__(self, neuron_type)

        self.image_shape = image_shape
        self.filter_shape = filter_shape

        assert image_shape[0] == filter_shape[1]

        self.init_parameters(filter_shape, (filter_shape[0],))

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

    def print_virtual(self):
        return "Image shape: {}\nFilter shape: {}\n".format(self.image_shape, self.filter_shape)


class LayerBlockConv2D(LayerBlockConv2DAbstract):
    """
    2D convolutional layer block
    """
    def __init__(self, neuron_type, image_shape, filter_shape):
        LayerBlockConv2DAbstract.__init__(self, neuron_type, image_shape, filter_shape)
        self.name = "2D convolutional layer block"

    def compute_bound_parameters_virtual(self):
        fan_in = np.prod(self.filter_shape[1:])
        fan_out = self.filter_shape[0] * np.prod(self.filter_shape[2:])

        return np.sqrt(6. / (fan_in + fan_out))

    def forward_virtual(self, conv_out):
        return self.neuron_type.activation_function(conv_out + self.b.dimshuffle('x', 0, 'x', 'x')).flatten(2)


class LayerBlockConvPool2D(LayerBlockConv2DAbstract):
    """
    2D convolutional layer + pooling layer. The reason for not having a separate pooling layer is that the combination
    of the two layers can be optimized.
    """
    def __init__(self, neuron_type, image_shape, filter_shape, poolsize=(2, 2)):
        self.poolsize = poolsize
        LayerBlockConv2DAbstract.__init__(self, neuron_type, image_shape, filter_shape)
        self.name = "2D convolutional + pooling layer"

    def compute_bound_parameters_virtual(self):
        fan_in = np.prod(self.filter_shape[1:])
        fan_out = (self.filter_shape[0] * np.prod(self.filter_shape[2:]) / np.prod(self.poolsize))

        return np.sqrt(6. / (fan_in + fan_out))

    def forward_virtual(self, conv_out):
        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=self.poolsize,
                                            ignore_border=True)

        return self.neuron_type.activation_function(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')).flatten(2)

    def print_virtual(self):
        return LayerBlockConv2DAbstract.print_virtual(self) + "Pool size: {}\n".format(self.poolsize)


class LayerBlockConvPool3D(LayerBlockOfNeurons):
    """
    3D convolutional layer + pooling layer
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
        LayerBlockOfNeurons.__init__(self, neuron_type)
        self.name = "3D convolutional + pooling layer block"

        in_width, in_height, in_depth = self.in_shape = in_shape
        flt_depth, flt_height, flt_width = self.flt_shape = flt_shape
        self.in_channels = in_channels
        self.flt_channels = flt_channels

        self.image_shape = (in_depth, in_channels, in_height, in_width)
        self.filter_shape = (flt_channels, flt_depth, in_channels, flt_height, flt_width)
        self.poolsize = poolsize

        self.init_parameters(self.filter_shape, (self.filter_shape[0],))

    def compute_bound_parameters_virtual(self):
        fan_in = np.prod(self.in_shape)
        fan_out = self.flt_channels * np.prod(self.flt_shape) / np.prod(self.poolsize)

        return np.sqrt(6. / (fan_in + fan_out))

    def forward(self, x, batch_size):
        img_batch_shape = (batch_size,) + self.image_shape

        x = x.reshape(img_batch_shape)

        # convolve input feature maps with filters
        conv_out = conv3d2d.conv3d(signals=x,
                                   filters=self.w,
                                   signals_shape=img_batch_shape,
                                   filters_shape=self.filter_shape,
                                   border_mode='valid')

        perm = [0, 2, 1, 3, 4]
        pooled_out = max_pool_3d(conv_out.dimshuffle(perm), self.poolsize, ignore_border=True)

        return self.neuron_type.activation_function(pooled_out.dimshuffle(perm)
                                                    + self.b.dimshuffle('x', 'x', 0, 'x', 'x')).flatten(2)

    def print_virtual(self):
        return "Image shape: {} \n Filter shape: {} \n Pool size: {} \n".format(
            self.image_shape, self.filter_shape, self.poolsize)
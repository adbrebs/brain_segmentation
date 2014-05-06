__author__ = 'adeb'

import theano
from theano import tensor as T

import layer


class Network():
    def __init__(self, n_in, n_out):
        print '... initialize the model'
        self.n_in = n_in
        self.n_out = n_out

        self.layers = []
        self.params = []

    def forward(self, x, batch_size):
        """Return the output of the network
        Args:
            x (theano.tensor.TensorType): input of the network
        Returns:
            (theano.tensor.TensorType): output of the network
        """
        y = x
        for l in self.layers:
            y = l.forward(y, batch_size)

        return y

    def generate_testing_function(self, batch_size):
        x = T.dmatrix('x')  # Minibatch input matrix
        y_pred = self.forward(x, batch_size)  # Output of the network
        return theano.function(x, y_pred)


class Network1(Network):
    def __init__(self, n_in, n_out):
        Network.__init__(self, n_in, n_out)

        self.layers.append(layer.LayerTan(n_in, 500))
        self.layers.append(layer.LayerSoftMax(500, n_out))

        self.params = []
        for l in self.layers:
            self.params += l.params


class Network2(Network):
    def __init__(self, patch_width, n_in, n_out):
        Network.__init__(self, n_in, n_out)

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
        # maxpooling reduces this further to (24/2,24/2) = (12,12)
        # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
        kernel_width0 = 5
        pool_size0 = 2
        n_kern0 = 20
        layer0 = layer.LayerConvPool2D(image_shape=(1, patch_width, patch_width),
                                       filter_shape=(n_kern0, 1, kernel_width0, kernel_width0),
                                       poolsize=(pool_size0, pool_size0))

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
        # maxpooling reduces this further to (8/2,8/2) = (4,4)
        # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
        filter_map_width1 = (patch_width - kernel_width0 + 1) / 2
        kernel_width1 = 5
        pool_size1 = 2
        n_kern1 = 50
        layer1 = layer.LayerConvPool2D(image_shape=(n_kern0, filter_map_width1, filter_map_width1),
                                       filter_shape=(n_kern1, n_kern0, kernel_width1, kernel_width1),
                                       poolsize=(pool_size1, pool_size1))

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (20,32*4*4) = (20,512)
        # construct a fully-connected sigmoidal layer
        filter_map_with2 = (filter_map_width1 - kernel_width1 + 1) / 2
        n_in2 = n_kern1 * filter_map_with2 * filter_map_with2
        n_out2 = 500
        layer2 = layer.LayerTan(n_in=n_in2, n_out=n_out2)

        # classify the values of the fully-connected sigmoidal layer
        layer3 = layer.LayerSoftMax(n_in=n_out2, n_out=self.n_out)

        self.layers = [layer0, layer1, layer2, layer3]

        self.params = []
        for l in self.layers:
            self.params += l.params
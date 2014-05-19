__author__ = 'adeb'

import sys
import h5py
import numpy as np

import theano
from theano import tensor as T

import layer
import neurons


class Network():
    """
    Attributes:
        n_in: List of pairs (mri_file, label_file)
        n_out: Number of output classes in the dataset

        layers(list): list of the layers composing the network
        params(list): list of arrays of parameters of all the layers

        scalar_mean: mean for standardizing the data
        scalar_std: std for standardizing the data
    """
    def __init__(self):
        self.n_in = None
        self.n_out = None

        self.layers = []
        self.params = []

        self.scalar_mean = None
        self.scalar_std = None

    def init(self, n_in, n_out):
        print '... initialize the model'
        self.n_in = n_in
        self.n_out = n_out

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
        """Generate a C-compiled function that can be used to compute the output of the network from an input batch
        Args:
            batch_size (int): the input of the returned function will be a batch of batch_size elements
        Returns:
            (function): function to returns the output of the network for a given input batch
        """
        x = T.matrix('x')  # Minibatch input matrix
        y_pred = self.forward(x, batch_size)  # Output of the network
        return theano.function([x], y_pred)

    def predict(self, patches):
        """Return the outputs of the patches
        Args:
            patches (2D array): dataset in which rows are datapoints
        Returns:
            pred (2D array): outputs of the network for the given patches
        """
        n_patches = patches.shape[0]
        pred = np.zeros((n_patches, self.n_out), dtype=np.float32)
        batch_size = min(200, n_patches)
        pred_fun = self.generate_testing_function(batch_size)

        n_batches = n_patches / batch_size
        n_rest = n_patches - n_batches*batch_size

        print "--------------------"
        for b in xrange(n_batches):
            sys.stdout.write("\rPrediction: %d%%" %((100*b)/n_batches))
            sys.stdout.flush()
            id0 = b*batch_size
            id1 = id0 + batch_size
            pred[id0:id1] = pred_fun(patches[id0:id1])

        if n_rest > 0:
            pred_fun_res = self.generate_testing_function(n_rest)
            pred[n_batches*batch_size:] = pred_fun_res(patches[n_batches*batch_size:])

        return pred

    def create_scaling_from_raw_data(self, data):
        self.scalar_mean = np.mean(data, axis=0)
        self.scalar_std = np.std(data, axis=0)

    def create_scaling_from_raw_database(self, ds):
        self.create_scaling_from_raw_data(ds.train_x.get_value(borrow=True))

    def scale_dataset(self, dataset):
        dataset.patch -= self.scalar_mean
        dataset.patch /= self.scalar_std

    def scale_database(self, database):
        database.train_x.set_value(self.scale_raw_data(database.train_x.get_value(borrow=True)))
        database.valid_x.set_value(self.scale_raw_data(database.valid_x.get_value(borrow=True)))
        database.test_x.set_value(self.scale_raw_data(database.test_x.get_value(borrow=True)))

    def scale_raw_data(self, data):
        data -= self.scalar_mean
        data /= self.scalar_std
        return data

    def save_parameters(self, file_name):
        """
        Save parameters (weights, biases) of the network in an hdf5 file
        """
        f = h5py.File("./networks/" + file_name, "w")
        f.attrs['n_in'] = self.n_in
        f.attrs['n_out'] = self.n_out
        self.save_parameters_virtual(f)
        for i, l in enumerate(self.layers):
            l.save_parameters(f, "layer" + str(i))
        f.create_dataset("scalar_mean", data=self.scalar_mean, dtype='f')
        f.create_dataset("scalar_std", data=self.scalar_std, dtype='f')
        f.close()

    def save_parameters_virtual(self, h5file):
        raise NotImplementedError

    def load_parameters(self, file_name):
        """
        Load parameters (weights, biases) of the network from an hdf5 file
        """
        f = h5py.File("./networks/" + file_name, "r")
        self.n_in = int(f.attrs["n_in"])
        self.n_out = int(f.attrs["n_out"])
        self.load_parameters_virtual(f)
        for i, l in enumerate(self.layers):
            l.load_parameters(f, "layer" + str(i))

        self.scalar_mean = f["scalar_mean"].value
        self.scalar_std = f["scalar_std"].value
        f.close()

    def load_parameters_virtual(self, h5file):
        raise NotImplementedError


class Network1(Network):
    def __init__(self):
        Network.__init__(self)

    def init(self, n_in, n_out):
        Network.init(self, n_in, n_out)

        self.layers.append(layer.LayerFullyConnected(neurons.NeuronRELU(), n_in, 100))
        self.layers.append(layer.LayerFullyConnected(neurons.NeuronSoftmax(), 100, n_out))

        self.params = []
        for l in self.layers:
            self.params += l.params

    def save_parameters_virtual(self, h5file):
        pass

    def load_parameters_virtual(self, h5file):
        self.init(self.n_in, self.n_out)


class Network2(Network):
    def __init__(self):
        Network.__init__(self)
        self.patch_width = None

    def init(self, patch_width, n_out):
        Network.init(self, patch_width*patch_width, n_out)

        self.patch_width = patch_width

        neuron_relu = neurons.NeuronRELU()

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
        # maxpooling reduces this further to (24/2,24/2) = (12,12)
        # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
        kernel_width0 = 5
        pool_size0 = 2
        n_kern0 = 20
        layer0 = layer.LayerConvPool2D(neuron_relu,
                                       image_shape=(1, patch_width, patch_width),
                                       filter_shape=(n_kern0, 1, kernel_width0, kernel_width0),
                                       poolsize=(pool_size0, pool_size0))

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
        # maxpooling reduces this further to (8/2,8/2) = (4,4)
        # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
        filter_map_width1 = (patch_width - kernel_width0 + 1) / pool_size0
        kernel_width1 = 5
        pool_size1 = 2
        n_kern1 = 50
        layer1 = layer.LayerConvPool2D(neuron_relu,
                                       image_shape=(n_kern0, filter_map_width1, filter_map_width1),
                                       filter_shape=(n_kern1, n_kern0, kernel_width1, kernel_width1),
                                       poolsize=(pool_size1, pool_size1))

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (20,32*4*4) = (20,512)
        # construct a fully-connected sigmoidal layer
        filter_map_with2 = (filter_map_width1 - kernel_width1 + 1) / pool_size1
        n_in2 = n_kern1 * filter_map_with2 * filter_map_with2
        n_out2 = 500
        layer2 = layer.LayerFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)

        # classify the values of the fully-connected sigmoidal layer
        layer3 = layer.LayerFullyConnected(neurons.NeuronSoftmax(), n_in=n_out2, n_out=self.n_out)

        self.layers = [layer0, layer1, layer2, layer3]

        self.params = []
        for l in self.layers:
            self.params += l.params

    def save_parameters_virtual(self, h5file):
        h5file.attrs['patch_width'] = self.patch_width

    def load_parameters_virtual(self, h5file):
        self.patch_width = int(h5file.attrs["patch_width"])
        self.init(self.patch_width, self.n_out)


class Network3(Network):
    def __init__(self):
        Network.__init__(self)
        self.patch_width = None

    def init(self, patch_width, n_out):
        Network.init(self, patch_width*patch_width, n_out)

        self.patch_width = patch_width

        neuron_relu = neurons.NeuronRELU()

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
        # maxpooling reduces this further to (24/2,24/2) = (12,12)
        # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
        kernel_width0 = 5
        n_kern0 = 10
        layer0 = layer.LayerConv2D(neuron_relu,
                                   image_shape=(1, patch_width, patch_width),
                                   filter_shape=(n_kern0, 1, kernel_width0, kernel_width0))

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
        # maxpooling reduces this further to (8/2,8/2) = (4,4)
        # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
        filter_map_width1 = (patch_width - kernel_width0 + 1)
        kernel_width1 = 5
        n_kern1 = 10
        layer1 = layer.LayerConv2D(neuron_relu,
                                   image_shape=(n_kern0, filter_map_width1, filter_map_width1),
                                   filter_shape=(n_kern1, n_kern0, kernel_width1, kernel_width1))

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (20,32*4*4) = (20,512)
        # construct a fully-connected sigmoidal layer
        filter_map_with2 = (filter_map_width1 - kernel_width1 + 1)
        n_in2 = n_kern1 * filter_map_with2 * filter_map_with2
        n_out2 = 100
        layer2 = layer.LayerFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)

        # classify the values of the fully-connected sigmoidal layer
        layer3 = layer.LayerFullyConnected(neurons.NeuronSoftmax(), n_in=n_out2, n_out=self.n_out)

        self.layers = [layer0, layer1, layer2, layer3]

        self.params = []
        for l in self.layers:
            self.params += l.params

    def save_parameters_virtual(self, h5file):
        h5file.attrs['patch_width'] = self.patch_width

    def load_parameters_virtual(self, h5file):
        self.patch_width = int(h5file.attrs["patch_width"])
        self.init(self.patch_width, self.n_out)
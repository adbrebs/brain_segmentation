__author__ = 'adeb'

import sys
import h5py
import numpy as np

import theano
from theano import tensor as T

import layer
import neurons


class Network(object):
    """Abstract class that needs to be inherited to define a specific network.
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

        self.scalar_mean = 0
        self.scalar_std = 0

    def init_common(self, n_in, n_out):
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

        # If ever one feature has the same value for all datapoints, the std will be zero and it will lead to some
        # divisions by zero. Therefore we set it to 1 in this case.

        self.scalar_std[self.scalar_std == 0] = 1

    def create_scaling_from_raw_database(self, ds):
        self.create_scaling_from_raw_data(ds.train_x.get_value(borrow=True))

    def scale_dataset(self, dataset):
        dataset.inputs -= self.scalar_mean
        dataset.inputs /= self.scalar_std

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
        f.attrs['network_type'] = self.__class__.__name__
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

    def __str__(self):
        n_parameters = 0
        for p in self.params:
            n_parameters += p.get_value().size
        msg = "This network has {} inputs, {} outputs and {} parameters \n".format(self.n_in, self.n_out, n_parameters)
        for i, l in enumerate(self.layers):
            msg += "------- Layer {} ------- \n".format(i)
            msg += l.__str__()
        return msg

    def export_params(self):
        params = []
        for p in self.params:
            params.append(p.get_value())
        return params

    def import_params(self, params):
        for p, p_sym in zip(params, self.params):
            p_sym.set_value(p)


class Network1(Network):
    def __init__(self):
        Network.__init__(self)

    def init(self, n_in, n_out):
        Network.init_common(self, n_in, n_out)

        neuron_type = neurons.NeuronRELU()
        self.layers.append(layer.LayerFullyConnected(neuron_type, n_in, 20))
        self.layers.append(layer.LayerFullyConnected(neuron_type, 20, 20))
        self.layers.append(layer.LayerFullyConnected(neurons.NeuronSoftmax(), 20, n_out))

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
        self.in_width = None
        self.in_height = None

    def init(self, patch_height, patch_width, n_out):
        Network.init_common(self, patch_height*patch_width, n_out)

        self.in_height = patch_height
        self.in_width = patch_width

        neuron_relu = neurons.NeuronRELU()

        # Layer 0
        kernel_height0 = 5
        kernel_width0 = 5
        pool_size_height0 = 2
        pool_size_width0 = 2
        n_kern0 = 20
        layer0 = layer.LayerConvPool2D(neuron_relu,
                                       image_shape=(1, patch_height, patch_width),
                                       filter_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                       poolsize=(pool_size_height0, pool_size_width0))

        # Layer 1
        filter_map_height1 = (patch_height - kernel_height0 + 1) / pool_size_height0
        filter_map_width1 = (patch_width - kernel_width0 + 1) / pool_size_width0
        kernel_height1 = 5
        kernel_width1 = 5
        pool_size_height1 = 2
        pool_size_width1 = 2
        n_kern1 = 50
        layer1 = layer.LayerConvPool2D(neuron_relu,
                                       image_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                       filter_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                       poolsize=(pool_size_height1, pool_size_width1))

        # Layer 2
        filter_map_height2 = (filter_map_height1 - kernel_height1 + 1) / pool_size_height1
        filter_map_with2 = (filter_map_width1 - kernel_width1 + 1) / pool_size_width1
        n_in2 = n_kern1 * filter_map_height2 * filter_map_with2
        n_out2 = 500
        layer2 = layer.LayerFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)

        # Layer 3
        layer3 = layer.LayerFullyConnected(neurons.NeuronSoftmax(), n_in=n_out2, n_out=self.n_out)

        self.layers = [layer0, layer1, layer2, layer3]

        self.params = []
        for l in self.layers:
            self.params += l.params

    def save_parameters_virtual(self, h5file):
        h5file.attrs['in_height'] = self.in_height
        h5file.attrs['in_width'] = self.in_width

    def load_parameters_virtual(self, h5file):
        self.in_height = int(h5file.attrs["in_height"])
        self.in_width = int(h5file.attrs["in_width"])
        self.init(self.in_height, self.in_width, self.n_out)


class Network3(Network):
    def __init__(self):
        Network.__init__(self)
        self.in_height = None
        self.in_width = None
        self.in_depth = None

    def init(self, patch_height, patch_width, patch_depth, n_out):
        Network.init_common(self, patch_height*patch_width*patch_depth, n_out)

        self.in_height = patch_height
        self.in_width = patch_width
        self.in_depth = patch_depth

        neuron_relu = neurons.NeuronRELU()

        # Layer 0
        filter_map_0_shape = np.array([patch_height, patch_width, patch_depth], dtype=int)
        filter_0_shape = np.array([2, 2, 2], dtype=int)
        pool_0_shape = np.array([2, 2, 2], dtype=int)
        n_kern0 = 20
        layer0 = layer.LayerConvPool3D(neuron_relu,
                                       1, tuple(filter_map_0_shape),
                                       n_kern0, tuple(filter_0_shape),
                                       poolsize=tuple(pool_0_shape))

        # Layer 1
        filter_map_1_shape = (filter_map_0_shape - filter_0_shape + 1) / pool_0_shape
        filter_1_shape = np.array([2, 2, 2], dtype=int)
        pool_1_shape = np.array([2, 2, 2], dtype=int)
        n_kern1 = 50
        layer1 = layer.LayerConvPool3D(neuron_relu,
                                       n_kern0, tuple(filter_map_1_shape),
                                       n_kern1, tuple(filter_1_shape),
                                       poolsize=tuple(pool_1_shape))

        # Layer 2
        filter_map_2_shape = (filter_map_1_shape - filter_1_shape + 1) / pool_1_shape
        n_in2 = n_kern1 * np.prod(filter_map_2_shape)
        n_out2 = 500
        layer2 = layer.LayerFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)

        # Layer 3
        layer3 = layer.LayerFullyConnected(neurons.NeuronSoftmax(), n_in=n_out2, n_out=self.n_out)

        self.layers = [layer0, layer1, layer2, layer3]

        self.params = []
        for l in self.layers:
            self.params += l.params

    def save_parameters_virtual(self, h5file):
        h5file.attrs['in_height'] = self.in_height
        h5file.attrs['in_width'] = self.in_width
        h5file.attrs['in_depth'] = self.in_depth

    def load_parameters_virtual(self, h5file):
        self.in_height = int(h5file.attrs["in_height"])
        self.in_width = int(h5file.attrs["in_width"])
        self.in_depth = int(h5file.attrs["in_depth"])
        self.init(self.in_height, self.in_width, self.in_depth, self.n_out)


def load_network(net_file):
    """
    Factory function to create a network from a network file
    """
    h5file = h5py.File("./networks/" + net_file, "r")
    network_type_str = h5file.attrs["network_type"]
    h5file.close()

    # Subclasses of Network
    network_type = [cls for cls in Network.__subclasses__() if cls.__name__ == network_type_str][0]

    net = network_type()
    net.load_parameters(net_file)

    return net
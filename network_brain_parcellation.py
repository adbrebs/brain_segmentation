__author__ = 'adeb'

from spynet.models.layer_block import *
import spynet.models.neuron_type as neuron_type
from spynet.models.layer import *
from spynet.models.network import Network


class NetworkUltimate(Network):
    def __init__(self):
        Network.__init__(self)
        self.in_width = None
        self.in_height = None

    def init(self, patch_height, patch_width, n_out):
        Network.init_common(self, patch_height*patch_width, n_out)

        self.in_height = patch_height
        self.in_width = patch_width

        self.layers = []
        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0
        l1 = patch_width ** 2
        l2 = l1 * 2
        l3 = l1 * 3
        l4 = l3 + 3
        self.layers.append(LayerDivide([0, l1, l2, l3, l4]))

        # Layer 1
        kernel_height0 = 5
        kernel_width0 = 5
        pool_size_height0 = 2
        pool_size_width0 = 2
        n_kern0 = 10
        block0 = LayerBlockConvPool2D(neuron_relu,
                                      image_shape=(1, patch_height, patch_width),
                                      filter_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      poolsize=(pool_size_height0, pool_size_width0))
        block1 = LayerBlockIdentity()

        self.layers.append(LayerOfBlocks([block0, block0, block0, block1]))

        # Layer 2
        filter_map_height1 = (patch_height - kernel_height0 + 1) / pool_size_height0
        filter_map_width1 = (patch_width - kernel_width0 + 1) / pool_size_width0
        kernel_height1 = 5
        kernel_width1 = 5
        pool_size_height1 = 2
        pool_size_width1 = 2
        n_kern1 = 20
        block0 = LayerBlockConvPool2D(neuron_relu,
                                      image_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      filter_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block1 = LayerBlockConvPool2D(neuron_relu,
                                      image_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      filter_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block2 = LayerBlockConvPool2D(neuron_relu,
                                      image_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      filter_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block3 = LayerBlockIdentity()

        self.layers.append(LayerOfBlocks([block0, block1, block2, block3]))

        # Layer 3
        self.layers.append(LayerMerge())

        # Layer 4
        filter_map_height2 = (filter_map_height1 - kernel_height1 + 1) / pool_size_height1
        filter_map_with2 = (filter_map_width1 - kernel_width1 + 1) / pool_size_width1
        n_in2 = 3 * n_kern1 * filter_map_height2 * filter_map_with2 + 3
        n_out2 = 500
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)
        self.layers.append(LayerOfBlocks([block0]))

        # Layer 5
        block0 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_in=n_out2, n_out=self.n_out)
        self.layers.append(LayerOfBlocks([block0]))

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
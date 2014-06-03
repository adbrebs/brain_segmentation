__author__ = 'adeb'

import numpy as np
import theano.tensor as T


class Layer():
    """
    This abstract class represents a layer of a neural network.
    """
    def __init__(self):
        self.params = []

    def forward(self, input_list, batch_size):
        """Return the output of the layer block
        Args:
            input_list (list of theano.tensor.TensorType): input of the block layer
        Returns:
            (list of theano.tensor.TensorType): output of the block layer
        """
        raise NotImplementedError

    def save_parameters(self, h5file, name):
        """
        Save all parameters of the block layer in a hdf5 file.
        """
        pass

    def load_parameters(self, h5file, name):
        """
        Load all parameters of the block layer in a hdf5 file.
        """
        pass

    def __str__(self):
        raise NotImplementedError


class LayerMerge(Layer):
    """
    Merge the outputs of the previous layer.
    """
    def __init__(self):
        Layer.__init__(self)

    def forward(self, input_list, batch_size):
        return [T.concatenate(list(input_list), axis=1)]

    def __str__(self):
        return "Merging layer\n"


class LayerDivide(Layer):
    """
    Divide the output of the previous layer so that different blocks can be used in the next layer.
    """
    def __init__(self, proportions):
        Layer.__init__(self)
        self.limits = proportions

    def forward(self, input_list, batch_size):
        if len(input_list) != 1:
            raise Exception("LayerDivide's input should be of length 1")
        input = input_list[0]

        output_list = []
        for i in xrange(len(self.limits) - 1):
            s = slice(self.limits[i], self.limits[i+1])
            output_list.append(input[:, s])
        return output_list

    def __str__(self):
        return "Dividing layer\n"


class LayerOfBlocks(Layer):
    """
    Layer composed of blocks.
    """
    def __init__(self, layer_blocks):
        Layer.__init__(self)
        self.layer_blocks = layer_blocks

        self.params = []
        for l in self.layer_blocks:
            self.params += l.params

    def forward(self, input_list, batch_size):
        output_list = []
        for x, layerBlock in zip(input_list, self.layer_blocks):
            output_list.append(layerBlock.forward(x, batch_size))
        return output_list

    def save_parameters(self, h5file, name):
        for i, l in enumerate(self.layer_blocks):
            l.save_parameters(h5file, name + "/block" + str(i))

    def load_parameters(self, h5file, name):
        for i, l in enumerate(self.layer_blocks):
            l.load_parameters(h5file, name + "/block" + str(i))

    def __str__(self):
        msg = "Layer composed of the following block(s):\n"
        for i, l in enumerate(self.layer_blocks):
            msg += "Block " + str(i) + ":\n" + l.__str__() + "\n"
        return msg


def convert_blocks_into_feed_forward_layers(ls_layer_blocks):
    """
    Convert a list of layer blocks into a list of LayerOfBlocks, each one containing a single block.
    """
    ls_layers = []
    for layer_block in ls_layer_blocks:
        ls_layers.append(LayerOfBlocks([layer_block]))
    return ls_layers
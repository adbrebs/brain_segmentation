__author__ = 'adeb'


from datetime import datetime

import h5py
import numpy as np

from utilities import open_h5file
from data_generator import DataGeneratorBrain


class Dataset():
    """
    Abstract class to create, store, save, load a dataset.

    Attributes:
        inputs (2D array): rows represent datapoints and columns represent features
        outputs (2D array): if known, corresponding outputs of the inputs
        n_in_features (int): number of input featurse
        n_out_features (int): number of output features
        n_data (int): number of datapoints
        is_perm (boolean): indicates if the dataset is shuffled or not
    """
    def __init__(self):
        self.inputs = None
        self.outputs = None

        self.n_in_features = None
        self.n_out_features = None
        self.n_data = None

        self.is_perm = False

    def populate_common(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

        self.n_data, self.n_in_features = inputs.shape
        self.n_out_features = outputs.shape[1]

    def shuffle_data(self):
        """
        Shuffle the dataset.
        """
        perm = np.random.permutation(self.n_data)
        self.inputs = self.inputs[perm]
        self.outputs = self.outputs[perm]
        self.shuffle_data_virtual(perm)

    def shuffle_data_virtual(self, perm):
        pass

    def write(self, file_name):
        """
        write the dataset in a hdf5 file.
        """
        h5file = h5py.File("./data/" + file_name, "w")
        h5file.create_dataset("inputs", data=self.inputs, dtype='f')
        h5file.create_dataset("outputs", data=self.outputs, dtype='f')

        h5file.attrs['creation_date'] = str(datetime.now())
        h5file.attrs['n_data'] = self.n_data
        h5file.attrs['n_in_features'] = self.n_in_features
        h5file.attrs['n_out_features'] = self.n_out_features
        h5file.attrs['is_perm'] = self.is_perm

        self.write_virtual(h5file)

        h5file.close()

    def write_virtual(self, h5file):
        pass

    def read(self, file_name):
        """
        load the dataset from a hdf5 file.
        """
        h5file = open_h5file("./data/" + file_name)
        self.inputs = h5file["inputs"].value
        self.outputs = h5file["outputs"].value

        self.n_data = int(h5file.attrs["n_data"])
        self.n_in_features = int(h5file.attrs["n_in_features"])
        self.n_out_features = int(h5file.attrs["n_out_features"])
        self.is_perm = bool(h5file.attrs['is_perm'])

        self.read_virtual(h5file)

        h5file.close()

    def read_virtual(self, h5file):
        pass


class DatasetBrainParcellation(Dataset):
    """
    Specialized dataset class for the brain parcellation data.
    Attributes:
        idx_patch(array n_patches x n_in_features): Array containing the indices of the voxels of the patches
        vx(array n_data x 3): Array containing the coordinates x, y, z of the voxels
        file_id(array n_data x 3): Array containing the file id of the datapoint
    """
    def __init__(self):
        Dataset.__init__(self)

        self.patch_width = None
        self.n_patch_per_voxel = 1

        # Initialize the additional containers
        self.idx_patch = None
        self.vx = None
        self.file_id = None

    def populate_from_config(self, config):
        data_generator = DataGeneratorBrain()
        data_generator.init_from_config(config)
        vx, inputs, idx_patch, outputs, file_id = data_generator.generate(config.getint('generate_data', 'n_data'))
        self.populate(inputs, outputs, vx, idx_patch, file_id, config.getint("pick_patch", "patch_width"))
        self.shuffle_data()

    def populate(self, inputs, outputs, vx, idx_patch, file_id, patch_width):
        Dataset.populate_common(self, inputs, outputs)
        self.vx = vx
        self.idx_patch = idx_patch
        self.file_id = file_id
        self.patch_width = patch_width

    def shuffle_data_virtual(self, perm):
        self.idx_patch = self.idx_patch[perm]
        self.vx = self.vx[perm]
        self.file_id = self.file_id[perm]

    def write_virtual(self, h5file):

        h5file.create_dataset("voxels", data=self.vx, dtype='f')
        h5file.create_dataset("idx_patches", data=self.idx_patch, dtype='f')
        h5file.create_dataset("file_id", data=self.file_id, dtype='f')

        h5file.attrs['n_patch_per_voxel'] = self.n_patch_per_voxel
        h5file.attrs['patch_width'] = self.patch_width

    def read_virtual(self, h5file):

        self.vx = h5file["voxels"].value
        self.idx_patch = h5file["idx_patches"].value
        self.file_id = h5file["file_id"].value

        self.n_patch_per_voxel = int(h5file.attrs["n_patch_per_voxel"])
        self.patch_width = int(h5file.attrs["patch_width"])
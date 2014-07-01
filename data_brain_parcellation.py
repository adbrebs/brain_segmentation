__author__ = 'adeb'

import os
import glob
import numpy as np
import nibabel as nib
import theano

from spynet.utils.utilities import distrib_balls_in_bins
from spynet.utils.multiprocess import parmap
from spynet.data.dataset import Dataset
from spynet.data.utils_3d.pick_patch import create_pick_patch
from spynet.data.utils_3d.pick_voxel import create_pick_voxel
from spynet.data.utils_3d.pick_target import create_pick_target


class DataGeneratorBrain():
    """
    Attributes:
        pick_vx(function): Function to pick voxels
        pick_patch(function): Function to pick patches
        pick_tg(function): Function to pick patches

        files: List of pairs (mri_file, label_file)
        atlases: List of pairs (mri array, label array)
        n_files (int): Number of files

        n_out_features (int): Number of output classes in the datasets
        n_patch_per_voxel (int): Number of patches per voxel
    """
    def __init__(self):

        self.pick_vx = None
        self.pick_patch = None
        self.pick_tg = None

        self.files = None
        self.n_files = None
        self.atlases = None

        self.n_out_features = None
        self.n_patch_per_voxel = None

    def init_from_config(self, config):
        self.pick_vx = create_pick_voxel(config)
        self.pick_patch = create_pick_patch(config)
        self.pick_tg = create_pick_target(config)
        self.files = list_miccai_files(**config.general["source_kwargs"])
        self.n_patch_per_voxel = config.general["n_patch_per_voxel"]
        self.__init_common()

    def init_from(self, files, pick_vx, pick_patch, pick_tg, n_patch_per_voxel):
        self.pick_vx = pick_vx
        self.pick_patch = pick_patch
        self.pick_tg = pick_tg
        self.files = files
        self.n_patch_per_voxel = n_patch_per_voxel
        self.__init_common()

    def __init_common(self):
        self.n_files = len(self.files)
        self.atlases = []
        for file_names in self.files:
            mri_file, lab_file = file_names
            mri = nib.load(mri_file).get_data().squeeze()
            lab = nib.load(lab_file).get_data().squeeze()
            mri, lab = crop_image(mri, lab)
            self.atlases.append((mri, lab))

        # Number of classes
        self.n_out_features = len(np.unique(self.atlases[0][1]))

    def generate(self, batch_size):
        print "Generate data ..."

        ### Initialization of the containers

        # Compute the number of voxels to extract from each atlas
        voxels_per_atlas = distrib_balls_in_bins(batch_size, self.n_files)

        # Initialize the containers
        vx = np.zeros((batch_size, 3), dtype=int)
        patch = np.zeros((batch_size, self.pick_patch.n_features), dtype=theano.config.floatX)
        idx_patch = np.zeros((batch_size, self.pick_patch.n_features), dtype=int)
        tg = np.zeros((batch_size, self.n_out_features), dtype=theano.config.floatX)
        file_id = np.zeros((batch_size, 1), dtype=int)

        ### Fill in the containers

        # Function that will be run in parallel
        def generate_from_one_brain(atlas_id):
            print("\r    file {}".format(self.files[atlas_id]))
            n_points = voxels_per_atlas[atlas_id]
            mri, lab = self.atlases[atlas_id]
            vx = self.pick_vx.pick(n_points, lab)
            patch, idx_patch = self.pick_patch.pick(vx, mri, lab)
            tg = self.pick_tg.pick(vx, idx_patch, self.n_out_features, mri, lab)

            return vx, patch, idx_patch, tg, atlas_id

        # Generate the patches in parallel
        if self.n_files == 1:  # This special case is necessary to avoid a bug on the server
            res_all = map(generate_from_one_brain, range(self.n_files))
        else:
            res_all = parmap(generate_from_one_brain, range(self.n_files))

        # Aggregate the data
        idx1 = 0
        for res in res_all:
            idx2 = idx1 + res[0].shape[0]
            vx[idx1:idx2], patch[idx1:idx2], idx_patch[idx1:idx2], tg[idx1:idx2], file_id[idx1:idx2] = res
            idx1 = idx2

        return vx, patch, idx_patch, tg, file_id


def list_miccai_files(**kwargs):
    """
    List the the pairs (mri_file_name, label_file_name) of the miccai data.
    """
    mode = kwargs["mode"]
    if mode == "miccai_challenge":
        source_folder = kwargs["source_folder"]
        mri_files = glob.glob("./datasets/miccai/" + source_folder + "/mri/*.nii")
        mri_files.sort()
        n_files = len(mri_files)
        label_path = "./datasets/miccai/" + source_folder + "/label/"

        return [(mri_files[i], label_path + os.path.basename(mri_files[i]))
                for i in xrange(0, n_files)]
    elif mode == "idx_files":
        idx_files = kwargs["idx_files"]
        mri_files = glob.glob("./datasets/miccai/1/mri/*.nii")
        n_files = len(mri_files)
        label_path = "./datasets/miccai/1/label/"

        list1 = [(mri_files[i], label_path + os.path.basename(mri_files[i]))
                 for i in xrange(0, n_files)]

        mri_files = glob.glob("./datasets/miccai/2/mri/*.nii")
        n_files = len(mri_files)
        label_path = "./datasets/miccai/2/label/"

        list2 = [(mri_files[i], label_path + os.path.basename(mri_files[i]))
                 for i in xrange(0, n_files)]

        list3 = list1 + list2
        list3.sort()

        return [list3[i] for i in idx_files]


def crop_image(mri, lab):
    """
    Extract the brain from an mri image
    """
    def check_limits(img):
        """
        Find the limits of the brain in an image
        """
        def check_limit_one_side(fun, iterations):
            for i in iterations:
                if np.any(fun(i)):
                    return i
            return iterations[-1]

        lim = np.zeros((3, 2), dtype=int)
        dims = img.shape

        f0 = lambda i: img[i, :, :]
        f1 = lambda i: img[:, i, :]
        f2 = lambda i: img[:, :, i]
        f = (f0, f1, f2)

        for j in xrange(3):
            lim[j, 0] = check_limit_one_side(f[j], xrange(dims[j]))
            lim[j, 1] = check_limit_one_side(f[j], reversed(xrange(dims[j])))

        return lim

    lim = check_limits(mri)
    lim0 = slice(lim[0, 0], lim[0, 1])
    lim1 = slice(lim[1, 0], lim[1, 1])
    lim2 = slice(lim[2, 0], lim[2, 1])

    mri = mri[lim0, lim1, lim2]
    lab = lab[lim0, lim1, lim2]

    return mri, lab


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
        vx, inputs, idx_patch, outputs, file_id = data_generator.generate(config.general["n_data"])
        self.populate(inputs, outputs, vx, idx_patch, file_id, config.pick_patch["patch_width"])
        self.shuffle_data()

    def populate(self, inputs, outputs, vx, idx_patch, file_id, patch_width):
        self.inputs = inputs
        self.outputs = outputs
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

    def duplicate_dataset_slice_virtual(self, ds, slice_idx):
        ds.vx = self.vx[slice_idx]
        ds.idx_patch = self.idx_patch[slice_idx]
        ds.file_id = self.file_id
        ds.patch_width = self.patch_width
        pass


def generate_and_save(config):
    file_path = config.general["file_path"]
    ds = DatasetBrainParcellation()
    ds.populate_from_config(config)
    ds.write(file_path)
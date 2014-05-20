__author__ = 'adeb'

import os
import glob
import math
from datetime import datetime

import h5py
import nibabel as nib

from multiprocess import parmap
from pick_voxel import *
from pick_patch import *
from pick_target import *


class Dataset():
    """
    Create, store, save, load a dataset.

    Attributes:
        inputs (2D array):
        outputs (2D array):
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

        self.is_perm = None

    def permute_data(self):
        perm = np.random.permutation(self.n_data)
        self.inputs = self.inputs[perm]
        self.outputs = self.outputs[perm]
        self.permute_data_virtual(perm)

    def permute_data_virtual(self, perm):
        pass

    def write(self, file_name):
        """
        write the dataset in a hdf5 file
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
        load the dataset from a hdf5 file
        """
        h5file = h5py.File("./data/" + file_name, "r")
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
    Attributes:
        file_list: List of pairs (mri_file, label_file)
        patch_width: Size of the 2D patch
        n_patch_per_voxel: There might be several different patches for each voxel
        pick_vx(function): Function to pick voxels
        pick_patch(function): Function to pick patches
        pick_tg(function): Function to pick patches
        idx_patch(array n_patches x patch_width^2): Array containing the indices of the voxels of the patches
        vx(array n_patches x 3): Array containing the coordinates x, y, z of the voxels
    """
    def __init__(self):
        Dataset.__init__(self)

        self.file_list = None
        self.n_files = None

        self.patch_width = None
        self.n_patch_per_voxel = None

        self.pick_vx = None
        self.pick_patch = None
        self.pick_tg = None

        # Initialize the containers
        self.idx_patch = None
        self.vx = None

    def permute_data_virtual(self, perm):
        self.idx_patch = self.idx_patch[perm]
        self.vx = self.vx[perm]

    def generate_from_config(self, config_ini):
        """
        Generate a new dataset from a config file
        """
        print '... generate the dataset'
        cat_ini = 'generate_data'

        # Collect the file names
        source = config_ini.get(cat_ini, 'source')
        if source == 'miccai':
            self.file_list = list_miccai_files()
            self.n_out_features = 139
        else:
            print "error source"

        self.n_files = len(self.file_list)
        self.patch_width = config_ini.getint("pick_tg", 'patch_width')
        self.n_patch_per_voxel = config_ini.getint(cat_ini, 'n_patch_per_voxel')

        # Create the objects responsible for picking the voxels
        self.pick_vx = self.create_pick_voxel(config_ini)
        self.pick_patch = self.create_pick_patch(config_ini)
        self.pick_tg = self.create_pick_target(config_ini)

        self.is_perm = config_ini.getboolean(cat_ini, 'perm')

        self.__generate_common()

    def generate_from(self, file_list, n_classes, patch_width, is_perm, n_patch_per_voxel, pick_vx, pick_patch, pick_tg):
        print '... generate the dataset'
        cat_ini = 'generate_data'

        self.file_list = file_list
        self.n_out_features = n_classes

        self.n_files = len(self.file_list)
        self.patch_width = patch_width
        self.n_patch_per_voxel = n_patch_per_voxel

        # Create the objects responsible for picking the voxels
        self.pick_vx = pick_vx
        self.pick_patch = pick_patch
        self.pick_tg = pick_tg

        self.is_perm = is_perm

        self.__generate_common()
        
    def __generate_common(self):
        """
        This private function is used in all the generation methods
        """

        # Create the mri-patch converter
        conv_mri_patch = ConverterMriPatch(self.patch_width, self.pick_vx, self.pick_patch, self.pick_tg)
        
        def extract_patches_from_file(i):
            mri_file, label_file = self.file_list[i]
            return conv_mri_patch.convert(mri_file, label_file, self.n_out_features)

        # Generation of the data in parallel on the CPU cores
        res_all = parmap(extract_patches_from_file, range(self.n_files))

        # Number of input features
        self.n_in_features = res_all[0][1].shape[1]

        # Count the number of patches
        self.n_data = 0
        for res in res_all:
            self.n_data += res[0].shape[0]
            
        # Initialize the containers
        self.inputs = np.zeros((self.n_data, self.n_in_features), dtype=np.float32)
        self.idx_patch = np.zeros((self.n_data, self.n_in_features), dtype=int)
        self.vx = np.zeros((self.n_data, 3), dtype=int)
        self.outputs = np.zeros((self.n_data, self.n_out_features), dtype=np.float32)

        # Populate the containers
        idx_writing_1 = 0
        for res in res_all:
            idx_writing_2 = idx_writing_1 + res[0].shape[0]
            s = slice(idx_writing_1, idx_writing_2)
            self.vx[s], self.inputs[s], self.idx_patch[s], self.outputs[s] = res[0:4]
            idx_writing_1 = idx_writing_2

        # Permute data
        if self.is_perm:
            self.permute_data()

    def create_pick_voxel(self, config_ini):
        """
        Create the objects responsible for picking the voxels
        """

        where_vx = config_ini.get("pick_voxel", 'where')
        how_vx = config_ini.get("pick_voxel", 'how')
        if where_vx == "anywhere":
            select_region = SelectWholeBrain()
        elif where_vx == "plane":
            select_region = SelectPlaneXZ(100)
        else:
            print "error in pick_voxel"
            return

        if how_vx == "all":
            extract_voxel = ExtractVoxelAll(self.n_patch_per_voxel)
        else:
            n_vx = config_ini.getint('pick_voxel', 'n_vx')

            # Re-adjust so there is no rounding problem with the number of files and classes
            divisor = self.n_files * self.n_out_features
            n_vx = int(math.ceil(float(n_vx) / divisor) * divisor)
            n_vx_per_file = n_vx / self.n_files

            if how_vx == "random":
                extract_voxel = ExtractVoxelRandomly(self.n_patch_per_voxel, n_vx_per_file)
            elif how_vx == "balanced":
                extract_voxel = ExtractVoxelBalanced(self.n_patch_per_voxel, n_vx_per_file)
            else:
                print "error in pick_voxel"
                return

        return PickVoxel(select_region, extract_voxel)

    def create_pick_patch(self, config_ini):
        """
        Create the objects responsible for picking the patches
        """
        how_patch = config_ini.get("pick_patch", 'how')
        if how_patch == "3D":
            pick_patch = PickPatch3DSimple()
        elif how_patch == "2Dortho":
            axis = config_ini.getint("pick_patch", 'axis')
            pick_patch = PickPatchParallelOrthogonal(axis)
        else:
            print "error in pick_patch"
            return

        return pick_patch

    def create_pick_target(self, config_ini):
        """
        Create the objects responsible for picking the targets
        """
        how_tg = config_ini.get("pick_tg", 'how')
        if how_tg == "center":
            pick_tg = PickTgCentered()
        elif how_tg == "proportion":
            pick_tg = PickTgProportion()
        else:
            print "error in pick_tg"
            return

        return pick_tg

    def write_virtual(self, h5file):

        h5file.create_dataset("voxels", data=self.vx, dtype='f')
        h5file.create_dataset("idx_patches", data=self.idx_patch, dtype='f')

        h5file.attrs['n_patch_per_voxel'] = self.n_patch_per_voxel
        h5file.attrs['patch_width'] = self.patch_width

    def read_virtual(self, h5file):

        self.vx = h5file["voxels"].value
        self.idx_patch = h5file["idx_patches"].value

        self.n_patch_per_voxel = int(h5file.attrs["n_patch_per_voxel"])
        self.patch_width = int(h5file.attrs["patch_width"])


class ExtractPatchesFromFile(object):
    """
    Class used for parallelizing the extraction of patches in the different files
    """
    def __init__(self, file_list, n_classes, conv_mri_patch):
        self.file_list = file_list
        self.n_classes = n_classes
        self.conv_mri_patch = conv_mri_patch

    def __call__(self, i):
        mri_file, label_file = self.file_list[i]
        return self.conv_mri_patch.convert(mri_file, label_file, self.n_classes)


class ConverterMriPatch():
    """
    Class that manages the convertion of an mri file into a dataset of patches
    """
    def __init__(self, patch_width, pick_vx, pick_patch, pick_tg):
        self.patch_width = patch_width

        self.pick_vx = pick_vx
        self.pick_patch = pick_patch
        self.pick_tg = pick_tg

    def convert(self, mri_file, label_file, n_classes):
        print "    ... converting " + mri_file
        mri = nib.load(mri_file).get_data().squeeze()
        lab = nib.load(label_file).get_data().squeeze()
        mri, lab = crop_image(mri, lab)

        vx = self.pick_vx.pick(lab)
        patch, idx_patch = self.pick_patch.pick(vx, mri, lab, self.patch_width)
        tg = self.pick_tg.pick(vx, idx_patch, n_classes, mri, lab)

        return vx, patch, idx_patch, tg, mri, lab


def list_miccai_files():
    mri_files = glob.glob("./data/miccai/mri/*.nii")
    n_files = len(mri_files)
    label_path = "./data/miccai/label/"

    return [(mri_files[i], label_path + os.path.basename(mri_files[i]))
            for i in xrange(1, n_files)] # On purpose, don't include the first file (will be used for testing)


def crop_image(mri, lab):
    """
    Extract the brain from an mri image
    """
    lim = check_limits(mri)
    lim0 = slice(lim[0, 0], lim[0, 1])
    lim1 = slice(lim[1, 0], lim[1, 1])
    lim2 = slice(lim[2, 0], lim[2, 1])

    mri = mri[lim0, lim1, lim2]
    lab = lab[lim0, lim1, lim2]

    return mri, lab


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


def create_img_from_pred(vx, pred, shape):
    """
    Create a labelled image array from voxels and their predictions
    """
    pred_img = np.zeros(shape, dtype=np.uint8)
    if len(shape) == 2:
        pred_img[vx[:, 0], vx[:, 1]] = pred
    elif len(shape) == 3:
        pred_img[vx[:, 0], vx[:, 1], vx[:, 2]] = pred
    return pred_img


def analyse_data(targets):
    # Number of classes
    targets_scalar = np.argmax(targets, axis=1)
    classes = np.unique(targets_scalar)
    n_classes = len(classes)
    print("There are {} regions in the dataset".format(n_classes))

    a = np.bincount(targets_scalar)
    b = np.nonzero(a)[0]
    c = a[b].astype(float, copy=False)
    c /= sum(c)

    print("    The largest region represents {} % of the image".format(max(c) * 100))

    return b, c


def compute_dice(img_pred, img_true, n_classes_max):
    classes = np.unique(img_true)
    n_classes = len(classes)
    dices = np.zeros((n_classes_max, 1))

    for c in classes:
        class_pred = img_pred == c
        class_true = img_true == c
        class_common = class_pred[class_true]
        dices[c] = 2 * np.sum(class_common, dtype=float) / (np.sum(class_pred) + np.sum(class_true))

    return dices


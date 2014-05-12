__author__ = 'adeb'

import os
import sys
import glob
import math
import ConfigParser
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
        file_list: List of pairs (mri_file, label_file)
        n_classes: Number of output classes in the dataset
        patch_width: Size of the 2D patch
        n_vx: Number of different voxels in the dataset
        n_patch_per_voxel: There might be several different patches for each voxel
        pick_vx(function): Function to pick voxels
        pick_patch(function): Function to pick patches
        pick_tg(function): Function to pick patches
        is_perm(boolean): True if the data is unordered
        patch(array n_patches x patch_width^2): Array containing the patches
        idx_patch(array n_patches x patch_width^2): Array containing the indices of the voxels of the patches
        vx(array n_patches x 3): Array containing the coordinates x, y, z of the voxels
        tg(array n_patches x n_classes): Array containing the targets for each patch
    """
    def __init__(self):

        self.file_list = None
        self.n_files = None
        self.n_classes = None

        self.patch_width = None
        self.n_vx = None
        self.n_patch_per_voxel = None

        self.pick_vx = None
        self.pick_patch = None
        self.pick_tg = None

        self.n_vx_per_file = None
        self.n_patches = None
        self.n_patches_per_file = None

        # Initialize the containers
        self.is_perm = None
        self.patch = None
        self.idx_patch = None
        self.vx = None
        self.tg = None

    def generate(self, config_ini):
        """
        Generate a new dataset from a config file
        """
        print '... generate the dataset'
        cat_ini = 'generate_data'

        # Collect the file names
        source = config_ini.get(cat_ini, 'source')
        if source == 'miccai':
            self.file_list = list_miccai_files()
            self.n_classes = 139
        else:
            print "error source"

        self.n_files = len(self.file_list)
        self.patch_width = config_ini.getint(cat_ini, 'patch_width')
        self.n_vx = config_ini.getint(cat_ini, 'n_vx')
        self.n_patch_per_voxel = config_ini.getint(cat_ini, 'n_patch_per_voxel')

        # Re-adjust so there is no rounding problem with the number of files and classes
        divisor = self.n_files * self.n_classes
        self.n_vx = int(math.ceil(float(self.n_vx) / divisor) * divisor)
        self.n_vx_per_file = self.n_vx / self.n_files
        self.n_patches = self.n_vx * self.n_patch_per_voxel
        self.n_patches_per_file = self.n_patches / self.n_files

        self.patch = np.zeros((self.n_patches, self.patch_width**2))
        self.idx_patch = np.zeros((self.n_patches, self.patch_width**2), dtype=int)
        self.vx = np.zeros((self.n_patches, 3), dtype=int)
        self.tg = np.zeros((self.n_patches, self.n_classes))

        # Create the objects responsible for picking the voxels
        self.pick_vx = self.create_pick_voxel(config_ini)
        self.pick_patch = PickPatchParallelOrthogonal(1)
        self.pick_tg = PickTgCentered()

        # Create the mri-patch converter
        conv_mri_patch = ConverterMriPatch(self.patch_width, self.pick_vx, self.pick_patch, self.pick_tg)

        # Extract patches
        print '... fill the dataset'
        def extractPatchesFromFile(i):
            mri_file, label_file = self.file_list[i]
            return conv_mri_patch.convert(mri_file, label_file, self.n_classes)
        for i, res in enumerate(parmap(extractPatchesFromFile, range(self.n_files))):
            id0 = i * self.n_patches_per_file
            id1 = id0 + self.n_patches_per_file
            self.vx[id0:id1], self.patch[id0:id1], self.idx_patch[id0:id1], self.tg[id0:id1] = res[0:4]

        # Permute data
        self.is_perm = config_ini.getboolean(cat_ini, 'perm')
        if self.is_perm:
            perm = np.random.permutation(self.n_patches)
            self.patch = self.patch[perm]
            self.idx_patch = self.idx_patch[perm]
            self.vx = self.vx[perm]
            self.tg = self.tg[perm]

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
        elif how_vx == "random":
            extract_voxel = ExtractVoxelRandomly(self.n_patch_per_voxel, self.n_vx_per_file)
        elif how_vx == "balanced":
            extract_voxel = ExtractVoxelBalanced(self.n_patch_per_voxel, self.n_vx_per_file)
        else:
            print "error in pick_voxel"
            return

        return PickVoxel(select_region, extract_voxel)

    def write(self, file_name):

        f = h5py.File("../data/" + file_name, "w")
        f.create_dataset("patches", data=self.patch, dtype='f')
        f.create_dataset("targets", data=self.tg, dtype='f')
        f.create_dataset("voxels", data=self.vx, dtype='f')
        f.create_dataset("idx_patches", data=self.idx_patch, dtype='f')

        f.attrs['creation_date'] = str(datetime.now())
        f.attrs['n_vx'] = self.n_vx
        f.attrs['n_patch_per_voxel'] = self.n_patch_per_voxel
        f.attrs['n_patches'] = self.n_patches
        f.attrs['patch_width'] = self.patch_width
        f.attrs['n_classes'] = self.n_classes
        f.attrs['is_perm'] = self.is_perm
        f.close()

    def read(self, file_name):

        f = h5py.File("../data/" + file_name, "r")
        self.patch = f["patches"].value
        self.tg = f["targets"].value
        self.vx = f["voxels"].value
        self.idx_patch = f["idx_patches"].value

        self.n_vx = int(f.attrs["n_vx"])
        self.n_patch_per_voxel = int(f.attrs["n_patch_per_voxel"])
        self.n_patches = int(f.attrs["n_patches"])
        self.patch_width = int(f.attrs["patch_width"])
        self.n_classes = int(f.attrs["n_classes"])
        self.is_perm = bool(f.attrs['is_perm'])
        f.close()


class ConverterMriPatch():
    """
    mri file --> dataset of patches converter
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
    mri_files = glob.glob("../data/miccai/mri/*.nii")
    n_files = len(mri_files)
    label_path = "../data/miccai/label/"

    return [(mri_files[i], label_path + os.path.basename(mri_files[i]))
            for i in xrange(n_files)]


def check_limits(img):
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


def create_img_from_pred(vx, pred, shape):
    pred_img = np.zeros(shape, dtype=np.uint8)
    if len(shape) == 2:
        pred_img[vx[:, 0], vx[:, 1]] = pred
    elif len(shape) == 3:
        pred_img[vx[:, 0], vx[:, 1], vx[:, 2]] = pred
    return pred_img
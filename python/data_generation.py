__author__ = 'adeb'

import os
import sys
import glob
import math
import ConfigParser
from datetime import datetime

import h5py
import nibabel as nib

from pick_voxel import *
from pick_patch import *
from pick_target import *


class Dataset():
    def __init__(self):

        self.file_list = None
        self.n_classes = None
        self.n_files = None

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

        cat_ini = 'generate_data'

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

        self.pick_vx = PickVoxel("plane", "random")
        self.pick_patch = PickPatchParallelXZ()
        self.pick_tg = PickTgProportion()

        # Re-adjust
        divisor = self.n_files * self.n_classes
        self.n_vx = int(math.ceil(float(self.n_vx) / divisor) * divisor)
        self.n_vx_per_file = self.n_vx / self.n_files
        self.n_patches = self.n_vx * self.n_patch_per_voxel
        self.n_patches_per_file = self.n_patches / self.n_files

        self.patch = np.zeros((self.n_patches, self.patch_width**2))
        self.idx_patch = np.zeros((self.n_patches, self.patch_width**2), dtype=int)
        self.vx = np.zeros((self.n_patches, 3), dtype=int)
        self.tg = np.zeros((self.n_patches, self.n_classes))

        conv_mri_patch = ConverterMriPatch(self.n_vx_per_file, self.n_patch_per_voxel, self.patch_width,
                                           self.pick_vx, self.pick_patch, self.pick_tg)
        # Extract patches
        for i in xrange(self.n_files):
            id0 = i * self.n_patches_per_file
            id1 = id0 + self.n_patches_per_file
            mri_file, label_file = self.file_list[i]
            print mri_file
            mri = nib.load(mri_file).get_data().squeeze()
            lab = nib.load(label_file).get_data().squeeze()
            conv_mri_patch.convert(mri, lab,
                                   self.vx[id0:id1],
                                   self.patch[id0:id1],
                                   self.idx_patch[id0:id1],
                                   self.tg[id0:id1])

        # Permute data
        self.is_perm = config_ini.getboolean(cat_ini, 'perm')
        if self.is_perm:
            perm = np.random.permutation(self.n_patches)
            self.patch = self.patch[perm]
            self.idx_patch = self.idx_patch[perm]
            self.vx = self.vx[perm]
            self.tg = self.tg[perm]



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
    Convert a mri file into patches
    """
    def __init__(self, n_vx, n_repeat, patch_width, pick_vx, pick_patch, pick_tg):
        self.n_vx = n_vx
        self.n_repeat = n_repeat
        self.patch_width = patch_width

        self.pick_vx = pick_vx
        self.pick_patch = pick_patch
        self.pick_tg = pick_tg

    def convert(self, mri, lab, vx, patch, idx_patch, tg):
        mri, lab = crop_image(mri, lab)

        self.pick_vx.pick_voxel(vx, mri, lab, self.n_vx, self.n_repeat)
        self.pick_patch.pick_patch(idx_patch, patch, vx, mri, lab, self.patch_width)
        self.pick_tg.pick_target(tg, vx, idx_patch, mri, lab)


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

    lim = check_limits(mri)
    lim0 = slice(lim[0, 0], lim[0, 1])
    lim1 = slice(lim[1, 0], lim[1, 1])
    lim2 = slice(lim[2, 0], lim[2, 1])

    mri = mri[lim0, lim1, lim2]
    lab = lab[lim0, lim1, lim2]

    return mri, lab


def convert_whole_mri():

    mri_file = '../data/miccai/mri/1000.nii'
    label_file = '../data/miccai/mri/1000.nii'
    print mri_file
    mri = nib.load(mri_file).get_data().squeeze()
    lab = nib.load(label_file).get_data().squeeze()
    n_patches = len(lab.ravel().nonzeros())
    patch_width = 29
    n_classes = 139

    pick_vx = PickVoxel("plane", "balanced")
    pick_patch = PickPatchParallelXZ()
    pick_tg = PickTgProportion()

    patch = np.zeros((n_patches, patch_width**2))
    idx_patch = np.zeros((n_patches, patch_width**2), dtype=int)
    vx = np.zeros((n_patches, 3), dtype=int)
    tg = np.zeros((n_patches, n_classes))

    conv_mri_patch = ConverterMriPatch(patch_width, 1, patch_width, pick_vx, pick_patch, pick_tg)
    conv_mri_patch.convert()

    return patch, idx_patch, vx, tg


if __name__ == '__main__':
    if len(sys.argv) == 1:
        ini_file = "creation_training_data_1.ini"
    else:
        ini_file = str(sys.argv[1])

    # Load config
    config_ini = ConfigParser.ConfigParser()
    config_ini.read(ini_file)
    file_name = config_ini.get('generate_data', 'file_name')

    # Generate the dataset
    dc_training = Dataset()
    dc_training.generate(config_ini)
    dc_training.write(file_name)
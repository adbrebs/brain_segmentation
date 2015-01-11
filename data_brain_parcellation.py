__author__ = 'adeb'

import os
import glob
import time
import numpy as np
import nibabel as nib
import theano
import matplotlib.cm as cm
from matplotlib import pyplot as plt

from spynet.utils.utilities import distrib_balls_in_bins
from spynet.utils.multiprocess import parmap
from spynet.data.dataset import Dataset
from spynet.data.utils_3d.pick_patch import *
from spynet.data.utils_3d.pick_voxel import *
from spynet.data.utils_3d.pick_target import *

from spynet.utils.utilities import tile_raster_images
import PIL

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
    """

    # See Miccai rules
    ignored_labels = range(1,4)+range(5,11)+range(12,23)+range(24,30)+[33,34]+[42,43]+[53,54]+range(63,69)+[70,74]+\
                     range(80,100)+[110,111]+[126,127]+[130,131]+[158,159]+[188,189]

    true_labels = [4, 11, 23, 30, 31, 32, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 56, 57,
                   58, 59, 60, 61, 62, 69, 71, 72, 73, 75, 76, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 112,
                   113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 128, 129, 132, 133, 134, 135, 136,
                   137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156,
                   157, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178,
                   179, 180, 181, 182, 183, 184, 185, 186, 187, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200,
                   201, 202, 203, 204, 205, 206, 207]

    def __init__(self):

        self.pick_vx = None
        self.pick_features = None
        self.pick_tg = None

        self.files = None
        self.n_files = None
        self.atlases = []
        self.ls_region_centroids = []

        self.n_out_features = None

    def init_from_config(self, config):
        self.pick_vx = create_pick_voxel(config)
        self.pick_features = create_pick_features(config)
        self.pick_tg = create_pick_target(config)
        self.files = list_miccai_files(**config.general["source_kwargs"])
        self.__init_common()

    def init_from(self, files, pick_vx, pick_patch, pick_tg):
        self.pick_vx = pick_vx
        self.pick_features = pick_patch
        self.pick_tg = pick_tg
        self.files = files
        self.__init_common()

    def __init_common(self):
        self.n_files = len(self.files)
        self.ls_region_centroids = [None]*self.n_files

        print "    preprocess the atlases ..."
        for i, file_names in enumerate(self.files):
            mri_file, lab_file = file_names
            print "        " + mri_file
            # nib.nifti1.FLOAT32_EPS_3 = -1e-6
            mri = nib.load(mri_file).get_data().squeeze()
            mri = mri.astype(np.float32, copy=False)
            lab = nib.load(lab_file).get_data().squeeze()
            lab = lab.astype(np.int16, copy=False)

            mri, lab = crop_brain_and_pad(mri, lab, self.pick_features.required_pad)
            self.scale_atlas(mri, lab)
            # plt.imshow(mri[100,:,:], cmap = cm.Greys_r)
            # plt.savefig("salut1.png")
            affine = nib.load(mri_file).get_affine()
            self.atlases.append((mri, lab, affine))

            # Compute the centroids
            if self.pick_features.has_instance_of(PickCentroidDistances):
                region_centroids = RegionCentroids(134)
                temp = lab.nonzero()
                vxs = np.asarray(temp).T
                region_centroids.update_barycentres(vxs, lab[temp])
                self.ls_region_centroids[i] = region_centroids

        # Number of classes
        self.n_out_features = 135

    def scale_atlas(self, mri, label):
        only_brain = mri[label.nonzero()]
        scalar_mean = np.mean(only_brain)
        scalar_std = np.std(only_brain)
        mri -= scalar_mean
        mri /= scalar_std

    def generate_single_atlas(self, atlas_id, n_points, region_centroid, batch_size, verbose=False):

        print("    file {} \n".format(self.files[atlas_id]))

        mri, lab, _ = self.atlases[atlas_id]
        vx_batches_generator = self.pick_vx.pick(n_points, lab, verbose=verbose, batch_size=batch_size)
        for vx_batch in vx_batches_generator:
            patch = self.pick_features.pick(vx_batch, mri, lab, region_centroid)[0]

            # patch_lab = self.pick_features.pick(vx_batch, lab, lab, self.ls_region_centroids[atlas_id])[0]
            # image1_1 = PIL.Image.fromarray(tile_raster_images(X=patch[0:10],
            #                                                img_shape=(29, 29), tile_shape=(1, 10),
            #                                                tile_spacing=(1, 1)))
            # image1_1.save("patches_2D_mri_1.png")
            # patch_lab = self.pick_features.pick(vx_batch, lab, lab, self.ls_region_centroids[atlas_id])[0]
            # image1_1 = PIL.Image.fromarray(tile_raster_images(X=patch[10:20],
            #                                                img_shape=(29, 29), tile_shape=(1, 10),
            #                                                tile_spacing=(1, 1)))
            # image1_1.save("patches_2D_mri_2.png")
            # temp_arr = tile_raster_images(X=patch_lab[0:10],
            #                               img_shape=(29, 29), tile_shape=(1, 10),
            #                               tile_spacing=(1, 1))
            # image2_1 = PIL.Image.fromarray(np.uint8(cm.spectral(temp_arr)*255))
            # image2_1.save("patches_2D_seg_1.png")
            # temp_arr = tile_raster_images(X=patch_lab[10:20],
            #                               img_shape=(29, 29), tile_shape=(1, 10),
            #                               tile_spacing=(1, 1))
            # image2_1 = PIL.Image.fromarray(np.uint8(cm.spectral(temp_arr)*255))
            # image2_1.save("patches_2D_seg_2.png")

            tg = self.pick_tg.pick(vx_batch, self.n_out_features, mri, lab)
            yield vx_batch, patch, tg

    def generate_parallel(self, batch_size):
        print "Generate data ..."

        ### Initialization of the containers

        # Compute the number of voxels to extract from each atlas
        voxels_per_atlas = distrib_balls_in_bins(batch_size, self.n_files)

        ### Fill in the containers

        # Function that will be run in parallel
        def generate_from_one_brain(atlas_id):
            n_points = voxels_per_atlas[atlas_id]
            # Large batch_size so it can not be reached. We want to store everything, so we don't split
            vx, patch, tg = \
                next(self.generate_single_atlas(atlas_id, n_points,
                                                self.ls_region_centroids[atlas_id], batch_size=1000000))
            return vx, patch, tg, atlas_id

        # Generate the patches in parallel
        if self.n_files == 1:  # This special case is necessary to avoid a bug on the server
            res_all = map(generate_from_one_brain, range(self.n_files))
        else:
            res_all = parmap(generate_from_one_brain, range(self.n_files))


        # Initialize the containers
        vx = np.zeros((batch_size, 3), dtype=int)
        patch = np.zeros((batch_size, self.pick_features.n_features), dtype=theano.config.floatX)
        tg = np.zeros((batch_size, self.n_out_features), dtype=theano.config.floatX)
        file_id = np.zeros((batch_size, 1), dtype=int)

        # Aggregate the data
        idx1 = 0
        for res in res_all:
            idx2 = idx1 + res[0].shape[0]
            vx[idx1:idx2], patch[idx1:idx2], tg[idx1:idx2], file_id[idx1:idx2] = res
            idx1 = idx2

        return vx, patch, tg, file_id


def list_miccai_files(**kwargs):
    """
    List the the pairs (mri_file_name, label_file_name) of the miccai data.
    """
    mode = kwargs["mode"]
    path = kwargs["path"]
    label_path = path + "label/"
    mri_files = glob.glob(path + "mri/*.nii")

    if mode == "folder":
        idx_files = xrange(len(mri_files))
    elif mode == "idx_files":
        idx_files = kwargs["idx_files"]
    else:
        raise Exception("Error to list the MICCAI files, the mode does not exist.")

    return [(mri_files[i], label_path + os.path.splitext(os.path.basename(mri_files[i]))[0] + "_glm.nii")
            for i in idx_files]


def check_img_limits(img):
    """
    Find the boundaries of the non-zero region of the image
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


def crop_brain_and_pad(mri, lab, pad):
    """
    Extract the brain from an mri image
    """

    lim = check_img_limits(lab)
    lim[:, 0] -= pad
    lim[:, 1] += pad

    dim_orig = np.array(mri.shape)
    pad_inf = np.zeros((3,), dtype=int)
    too_low = lim[:, 0] < 0
    pad_inf[too_low] = -lim[too_low, 0]
    lim[too_low, 0] = 0

    pad_sup = np.zeros((3,), dtype=int)
    too_high = lim[:, 1] > dim_orig
    pad_sup[too_high] = lim[too_high, 1] - dim_orig[too_high]
    lim[too_high, 1] = dim_orig[too_high]

    lim0 = slice(lim[0, 0], lim[0, 1])
    lim1 = slice(lim[1, 0], lim[1, 1])
    lim2 = slice(lim[2, 0], lim[2, 1])

    mri = mri[lim0, lim1, lim2]
    lab = lab[lim0, lim1, lim2]

    mri = np.lib.pad(mri, zip(pad_inf, pad_sup), 'constant', constant_values=0)
    lab = np.lib.pad(lab, zip(pad_inf, pad_sup), 'constant', constant_values=0)

    return mri, lab


class DatasetBrainParcellation(Dataset):
    """
    Specialized dataset class for the brain parcellation data.
    Attributes:
        vx(array n_data x 3): Array containing the coordinates x, y, z of the voxels
        file_ids(array n_data x 3): Array containing the file id of the datapoint
    """
    def __init__(self):
        Dataset.__init__(self)

        # Initialize the additional containers
        self.vx = None
        self.file_ids = None

    def populate_from_config(self, config):
        data_generator = DataGeneratorBrain()
        data_generator.init_from_config(config)
        vx, inputs, outputs, file_ids = data_generator.generate_parallel(config.general["n_data"])
        self.populate(inputs, outputs, vx, file_ids)
        self.shuffle_data()

    def populate(self, inputs, outputs, vx, file_ids):
        self.inputs = inputs
        self.outputs = outputs
        self.vx = vx
        self.file_ids = file_ids

    def shuffle_data_virtual(self, perm):
        self.vx = self.vx[perm]
        self.file_ids = self.file_ids[perm]

    def write_virtual(self, h5file):
        h5file.create_dataset("voxels", data=self.vx, dtype='f')
        h5file.create_dataset("file_id", data=self.file_ids, dtype='f')

    def read_virtual(self, h5file):
        self.vx = h5file["voxels"].value
        self.file_ids = h5file["file_id"].value

    def duplicate_datapoints_slice_virtual(self, ds, slice_idx):
        ds.vx = self.vx[slice_idx]
        ds.file_ids = self.file_ids
        pass


class RegionCentroids():
    def __init__(self, n_regions):
        self.n_regions = n_regions
        self.barycentres = np.zeros((n_regions, 3))

    def update_barycentres(self, vxs, regions):
        self.barycentres = np.zeros((self.n_regions, 3))
        for i in xrange(self.n_regions):
            idxs = regions == i+1
            if vxs[idxs].size == 0:
                continue
            self.barycentres[i] = np.mean(vxs[idxs], axis=0)

        # For zero values (with no regions present), set them to the mean
        self.barycentres[self.barycentres == 0] = self.barycentres[self.barycentres != 0].mean()

    def compute_scaled_distances(self, vx):
        distances = np.linalg.norm(self.barycentres - vx, axis=1)
        return distances


def generate_and_save(config):
    file_path = config.general["file_path"]
    ds = DatasetBrainParcellation()
    ds.populate_from_config(config)
    ds.write(file_path)


            # for ignored_label in self.ignored_labels:
            #     lab[lab == ignored_label] = 0
            # for idx, label in enumerate(self.true_labels):
            #     lab[lab==label] = idx+1
            #
            # aa = nib.Nifti1Image(mri, nib.load(mri_file).get_affine())
            # nib.save(aa, mri_file)
            #
            # bb = nib.Nifti1Image(lab, nib.load(lab_file).get_affine())
            # nib.save(bb, lab_file)
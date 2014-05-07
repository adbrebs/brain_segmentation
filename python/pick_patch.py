__author__ = 'adeb'

import numpy as np


class PickPatch():
    """
    Manage the selection and extraction of patches in an mri image from their central voxels
    """
    def __init__(self):
        pass

    def pick(self, vx, mri, label, patch_width):
        n_vx = vx.shape[0]
        idx_patch = np.zeros((n_vx, patch_width**2), dtype=int)
        patch = np.zeros((n_vx, patch_width**2))
        self.pick_virtual(patch, idx_patch, vx, mri, label, patch_width)
        return patch, idx_patch

    def pick_virtual(self, patch, idx_patch, vx, mri, label, patch_width):
        raise NotImplementedError


class PickPatchParallelXZ(PickPatch):
    def __init__(self):
        PickPatch.__init__(self)

    def pick_virtual(self, patch, idx_patch, vx, mri, label, patch_width):
        dims = mri.shape
        radius = patch_width / 2

        def crop(j, voxel):
            v = np.arange(voxel[j] - radius, voxel[j] + radius + 1)
            v[v < 0] = 0
            v[v >= dims[j]] = dims[j]-1
            return v

        for i in xrange(idx_patch.shape[0]):
            voxel = vx[i]

            v0 = crop(0, voxel)
            v1 = voxel[1]
            v2 = crop(2, voxel)

            x, z = np.meshgrid(v0, v2)
            idx_patch[i] = np.ravel_multi_index((x.ravel(), np.tile(v1, x.size), z.ravel()), dims)
            patch[i] = mri[x,v1,z].ravel()
__author__ = 'adeb'

import numpy as np


class PickPatch():
    def __init__(self):
        pass

    def pick_patch(self, idx_patch, patch, vx, mri, label, patch_width):
        raise NotImplementedError


class PickPatchParallelXZ(PickPatch):
    def __init__(self):
        PickPatch.__init__(self)

    def pick_patch(self, idx_patch, patch, vx, mri, label, patch_width):
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

            x, y, z = np.meshgrid(v0, v1, v2)
            idx_patch[i] = np.ravel_multi_index((x, y, z), dims).flatten()
            patch[i] = mri[x, y, z].flatten()
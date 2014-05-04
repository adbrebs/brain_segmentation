__author__ = 'adeb'

import numpy as np


class PickPatch():
    def __init__(self, dataset):
        self.ds = dataset

    def pick_patch(self, id0, id1, mri, label):
        raise NotImplementedError


class PickPatchParallelXZ(PickPatch):
    def __init__(self, dataset):
        PickPatch.__init__(self, dataset)

    def pick_patch(self, id0, id1, mri, label):
        dims = mri.shape
        radius = self.ds.patch_width / 2

        def crop(j, voxel):
            v = np.arange(voxel[j] - radius, voxel[j] + radius + 1)
            v[v < 0] = 0
            v[v >= dims[j]] = dims[j]-1
            return v

        for i in xrange(id0, id1):
            vx = self.ds.vx[i]

            v0 = crop(0, vx)
            v1 = vx[1]
            v2 = crop(2, vx)

            x, y, z = np.meshgrid(v0, v1, v2)
            self.ds.idx_patch[i] = np.ravel_multi_index((x, y, z), dims).flatten()
            self.ds.patch[i] = mri[x, y, z].flatten()


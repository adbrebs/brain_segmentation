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


class PickPatchParallelOrthogonal(PickPatch):
    def __init__(self, parallel_axis):
        PickPatch.__init__(self)
        self.parallel_axis = parallel_axis

    def pick_virtual(self, patch, idx_patch, vx, mri, label, patch_width):
        dims = mri.shape
        radius = patch_width / 2

        def crop(j, voxel):
            v = np.arange(voxel[j] - radius, voxel[j] + radius + 1)
            v[v < 0] = 0
            v[v >= dims[j]] = dims[j]-1
            return v

        for i in xrange(idx_patch.shape[0]):
            vx_cur = vx[i]
            v_parallel_axis = vx_cur[self.parallel_axis]
            v_other_axis = []
            l = range(3)
            del l[self.parallel_axis]
            for ax in l:
                v_other_axis.append(crop(ax, vx_cur))

            x, y = np.meshgrid(v_other_axis[0], v_other_axis[1])
            idx_patch[i] = np.ravel_multi_index((x.ravel(), np.tile(v_parallel_axis, x.size), y.ravel()), dims)
            patch[i] = mri[x, v_parallel_axis, y].ravel()
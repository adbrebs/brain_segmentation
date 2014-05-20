__author__ = 'adeb'

import numpy as np

from scipy.ndimage.interpolation import rotate


class PickPatch():
    """
    Manage the selection and extraction of patches in an mri image from their central voxels
    """
    def __init__(self):
        pass

    def pick(self, vx, mri, label, patch_width):
        raise NotImplementedError


class PickPatch2D(PickPatch):
    def __init__(self):
        PickPatch.__init__(self)

    def pick(self, vx, mri, label, patch_width):
        n_vx = vx.shape[0]
        idx_patch = np.zeros((n_vx, patch_width**2), dtype=int)
        patch = np.zeros((n_vx, patch_width**2), dtype=np.float32)
        self.pick_virtual2d(patch, idx_patch, vx, mri, label, patch_width)
        return patch, idx_patch

    def pick_virtual2d(self, patch, idx_patch, vx, mri, label, patch_width):
        raise NotImplementedError


class PickPatchParallelOrthogonal(PickPatch2D):
    """
    Pick a 2D patch centered on the voxels. No rotation
    """
    def __init__(self, parallel_axis):
        PickPatch2D.__init__(self)
        self.parallel_axis = parallel_axis

    def pick_virtual2d(self, patch, idx_patch, vx, mri, label, patch_width):
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


class PickPatchSlightlyRotated(PickPatch2D):
    """
    Pick a 2D patch centered on the voxels. Brains are slightly rotated.
    """
    def __init__(self, parallel_axis, max_degree_rotation):
        PickPatch2D.__init__(self)
        self.parallel_axis = parallel_axis
        self.max_degree_rotation = max_degree_rotation

    def pick_virtual2d(self, patch, idx_patch, vx, mri, label, patch_width):
        dims = mri.shape
        radius = patch_width / 2

        def crop(j, voxel):
            li = voxel[j] - radius
            ls = voxel[j] + radius + 1
            if li < 0:
                li = 0
            if ls >= dims[j]:
                ls = dims[j]-1
            return slice(li, ls)

        for i in xrange(idx_patch.shape[0]):
            vx_cur = vx[i]
            v_axis = []
            for ax in range(3):
                v_axis.append(crop(ax, vx_cur))

            cube = mri[v_axis[0], v_axis[1], v_axis[2]]
            cube = rotate(cube, np.random.uniform(-self.max_degree_rotation, -self.max_degree_rotation), axes=(0, 1))
            cube = rotate(cube, np.random.uniform(-self.max_degree_rotation, -self.max_degree_rotation), axes=(1, 2))
            cube = rotate(cube, np.random.uniform(-self.max_degree_rotation, -self.max_degree_rotation), axes=(2, 0))

            central_vx_cube = np.array(cube.shape)/2
            li = central_vx_cube - radius
            ls = central_vx_cube + radius + 1
            li[self.parallel_axis] = central_vx_cube[self.parallel_axis]
            ls[self.parallel_axis] = central_vx_cube[self.parallel_axis]+1

            patch[i] = cube[li[0]:ls[0], li[1]:ls[1], li[2]:ls[2]].ravel()


class PickPatch3D(PickPatch):
    def __init__(self):
        PickPatch.__init__(self)

    def pick(self, vx, mri, label, patch_width):
        n_vx = vx.shape[0]
        idx_patch = np.zeros((n_vx, patch_width**3), dtype=int)
        patch = np.zeros((n_vx, patch_width**3), dtype=np.float32)
        self.pick_virtual3d(patch, idx_patch, vx, mri, label, patch_width)
        return patch, idx_patch

    def pick_virtual3d(self, patch, idx_patch, vx, mri, label, patch_width):
        raise NotImplementedError


class PickPatch3DSimple(PickPatch3D):
    """
    Pick 3D patches centered on the voxels. No rotation
    """
    def __init__(self):
        PickPatch.__init__(self)

    def pick_virtual3d(self, patch, idx_patch, vx, mri, label, patch_width):
        dims = mri.shape
        radius = patch_width / 2

        def crop(j, voxel):
            v = np.arange(voxel[j] - radius, voxel[j] + radius + 1)
            v[v < 0] = 0
            v[v >= dims[j]] = dims[j]-1
            return v

        for i in xrange(idx_patch.shape[0]):
            vx_cur = vx[i]
            v_axis = []
            for ax in range(3):
                v_axis.append(crop(ax, vx_cur))

            x, y, z = np.meshgrid(v_axis[0], v_axis[1], v_axis[2])
            idx_patch[i] = np.ravel_multi_index((x.ravel(), y.ravel(), z.ravel()), dims)
            patch[i] = mri[x, y, z].ravel()



        # def pick_virtual(self, patch, idx_patch, vx, mri, label, patch_width):
        # dims = mri.shape
        # radius = patch_width / 2
        #
        # img = np.zeros((patch_width, patch_width, patch_width))
        #
        # def check_lims(ax, voxel):
        #     padd_inf = 0
        #     padd_sup = 0
        #     l_inf = voxel - radius
        #     if l_inf < 0:
        #         padd_inf = - l_inf
        #         l_inf = 0
        #     l_sup = voxel + radius + 1
        #     if l_sup > dims[ax]:
        #         padd_sup = l_sup - dims[ax]
        #         l_sup = dims[ax]
        #     return slice(padd_inf, padd_sup), slice(l_inf, l_sup)
        #
        #
        # for i in xrange(idx_patch.shape[0]):
        #     vx_cur = vx[i]
        #     s1 = []
        #     s2 = []
        #     for ax in range(3):
        #         res1, res2 = check_lims(ax, vx_cur[ax])
        #         s1.append(res1)
        #         s2.append(res2)
        #
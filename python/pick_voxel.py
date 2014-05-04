__author__ = 'adeb'

import numpy as np


class PickVoxel():
    def __init__(self, dataset):
        self.ds = dataset

    def pick_voxel(self, id0, id1, mri, label):
        raise NotImplementedError

    def duplicate_index(self, id0, id1, shape, vx_idx):
        vx_idx = np.repeat(vx_idx, self.ds.n_patch_per_voxel)
        self.ds.vx[id0:id1] = np.asarray(np.unravel_index(vx_idx, shape)).T


class PickVxRandomly(PickVoxel):
    def __init__(self, dataset):
        PickVoxel.__init__(self, dataset)

    def pick_voxel(self, id0, id1, mri, label):
        in_brain = label.ravel().nonzero()[0]
        r = np.random.randint(in_brain.size, size=self.ds.n_vx_per_file)
        vx_idx = in_brain[r]
        self.duplicate_index(id0, id1, mri.shape, vx_idx)


class PickVxRandomlyInPlaneXZ(PickVoxel):
    def __init__(self, dataset, y):
        PickVoxel.__init__(self, dataset)
        self.y = y

    def pick_voxel(self, id0, id1, mri, label):
        plan = np.zeros(mri.shape)
        plan[:, self.y, :] = label[:, self.y, :].squeeze()
        in_brain = plan.ravel().nonzero()[0]
        r = np.random.randint(in_brain.size, size=self.ds.n_vx_per_file)
        vx_idx = in_brain[r]
        self.duplicate_index(id0, id1, mri.shape, vx_idx)


class PickVxBalanced(PickVoxel):
    def __init__(self, dataset):
        PickVoxel.__init__(self, dataset)

    def pick_voxel(self, id0, id1, mri, label):
        n_vx_per_class = self.ds.n_vx_per_file / self.ds.n_classes
        vx_idx = np.zeros((self.ds.n_vx_per_file, 1))

        for k in xrange(self.ds.n_classes):
            region = np.where(label.ravel() == k)
            r = np.random.randint(len(region), size=n_vx_per_class)
            vx_idx[k * n_vx_per_class:(k+1)*n_vx_per_class] = region.flat[r]

        self.duplicate_index(id0, id1, mri.shape, vx_idx)
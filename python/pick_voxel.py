__author__ = 'adeb'

import numpy as np


class PickVoxel():
    def __init__(self, dataset):
        self.ds = dataset

    def pick_voxel(self, id0, id1, mri, label):
        raise NotImplementedError


class PickVxRandomlyInPlane(PickVoxel):
    def __init__(self, dataset, plan):
        PickVoxel.__init__(self, dataset)
        self.plan = plan

    def pick_voxel(self, id0, id1, mri, label):
        y = 100
        plan = np.zeros(mri.shape)
        plan[:, y, :] = label[:, y, :].squeeze()
        in_brain = plan.ravel().nonzero()[0]
        r = np.random.randint(in_brain.size, size=self.ds.n_vx_per_file)
        vx_idx = in_brain[r]
        vx_idx = np.repeat(vx_idx, self.ds.n_patch_per_voxel)
        self.ds.vx[id0:id1] = np.asarray(np.unravel_index(vx_idx, mri.shape)).T
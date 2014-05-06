__author__ = 'adeb'

import numpy as np


class PickVoxel():
    def __init__(self, dataset, where, how, **kwargs):
        self.ds = dataset
        self.where = where
        self.how = how
        self.kwargs = kwargs

    def pick_voxel(self, id0, id1, mri, label):

        if self.where == "anywhere":
            idx_flat = label.ravel().nonzero()[0]
        elif self.where == "plane":
            y = 100
            plan = np.zeros(mri.shape)
            plan[:, y, :] = label[:, y, :].squeeze()
            idx_flat = plan.ravel().nonzero()[0]
        else:
            print "error in pick_voxel"
            return

        if self.how == "random":
            vx_idx = self.pick_random_voxel(idx_flat)
        elif self.how == "balanced":
            vx_idx = self.pick_balanced_voxel(idx_flat, label)
        else:
            print "error in pick_voxel"
            return

        self.duplicate_index(id0, id1, mri.shape, vx_idx)

    def duplicate_index(self, id0, id1, shape, vx_idx):
        vx_idx = np.repeat(vx_idx, self.ds.n_patch_per_voxel)
        self.ds.vx[id0:id1] = np.asarray(np.unravel_index(vx_idx, shape)).T

    def pick_random_voxel(self, idx_flat):
        r = np.random.randint(idx_flat.size, size=self.ds.n_vx_per_file)
        vx_idx = idx_flat[r]
        return vx_idx

    def pick_balanced_voxel(self, idx_flat, label):
        classes_present = np.unique(label.flat[idx_flat])
        n_classes_present = len(classes_present)
        n_vx_per_class = self.ds.n_vx_per_file / n_classes_present
        vx_idx = np.zeros((self.ds.n_vx_per_file,), dtype=int)

        for id_k, k in enumerate(classes_present):
            region = np.where(label.flat[idx_flat] == k)[0]
            r = np.random.randint(len(region), size=n_vx_per_class)
            vx_idx[id_k * n_vx_per_class:(id_k+1)*n_vx_per_class] = region[r]

        return vx_idx
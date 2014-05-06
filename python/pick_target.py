__author__ = 'adeb'

import numpy as np


class PickTarget():
    """
    Manage the labelling of the patches
    """
    def __init__(self):
        pass

    def pick(self, vx, idx_patch, n_classes, mri, label):
        tg = np.zeros((vx.shape[0], n_classes), dtype=int)
        self.pick_virtual(tg, vx, idx_patch, n_classes, mri, label)
        return tg

    def pick_virtual(self, tg, vx, idx_patch, n_classes, mri, label):
        raise NotImplementedError


class PickTgCentered(PickTarget):
    """
    The label of each patch is the label of the central voxel of the patch
    """
    def __init__(self):
        PickTarget.__init__(self)

    def pick_virtual(self, tg, vx, idx_patch, n_classes, mri, label):
        tg[np.arange(tg.shape[0]), label[[vx[:, i] for i in xrange(3)]]] = 1


class PickTgProportion(PickTarget):
    """
    For each patch, the target is the vector of proportions of each class in the patch
    """
    def __init__(self):
        PickTarget.__init__(self)

    def pick_virtual(self, tg, vx, idx_patch, n_classes, mri, label):
        lab_flat = label.ravel()
        for i in xrange(vx.shape[0]):
            a = np.bincount(lab_flat[idx_patch[i]])
            b = np.nonzero(a)[0]
            c = a[b].astype(float, copy=False)
            c = c / sum(c)
            tg[i, b] = c
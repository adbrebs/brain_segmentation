__author__ = 'adeb'

import numpy as np


class PickTarget():
    def __init__(self):
        pass

    def pick_target(self, tg, vx, idx_patch, mri, label):
        raise NotImplementedError


class PickTgCentered(PickTarget):
    def __init__(self):
        PickTarget.__init__(self)

    def pick_target(self, tg, vx, idx_patch, mri, label):
        tg[np.arange(tg.shape[0]), label[[vx[:, i] for i in xrange(3)]]] = 1


class PickTgProportion(PickTarget):
    def __init__(self):
        PickTarget.__init__(self)

    def pick_target(self, tg, vx, idx_patch, mri, label):
        lab_flat = label.ravel()
        for i in xrange(tg.shape[0]):
            a = np.bincount(lab_flat[idx_patch[i]])
            b = np.nonzero(a)[0]
            c = a[b].astype(float, copy=False)
            c = c / sum(c)
            tg[i, b] = c
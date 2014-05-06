__author__ = 'adeb'

import numpy as np


class PickVoxel():
    """
    Manage the selection and extraction of voxels in an mri image
    """
    def __init__(self, select_region, extract_voxel):
        self.select_region = select_region
        self.extract_voxel = extract_voxel

    def pick(self, label):
        # Select the region in which voxels are going to be extracted
        idx_region = self.select_region.select(label)
        region = label.ravel()[idx_region]
        # Once the region is selected, extract the voxels
        return self.extract_voxel.extract(idx_region, region, label.shape)


class SelectRegion():
    """
    Select a specific spatial region of the mri image in which voxels will later be extracted
    """
    def __init__(self):
        pass

    def select(self, label):
        raise NotImplementedError


class SelectWholeBrain(SelectRegion):
    def __init__(self):
        SelectRegion.__init__(self)

    def select(self, label):
        return label.ravel().nonzero()[0]


class SelectPlaneXZ(SelectRegion):
    def __init__(self, y):
        SelectRegion.__init__(self)
        self.y = y

    def select(self, label):
        plan = np.zeros(label.shape)
        plan[:, self.y, :] = label[:, self.y, :].squeeze()
        return plan.ravel().nonzero()[0]


class ExtractVoxel():
    """
    This class extract voxels from a given region of the mri image
    """
    def __init__(self, n_repeat):
        self.n_repeat = n_repeat

    def extract(self, idx_region, region, shape):
        vx_idx = self.extract_virtual(idx_region, region)

        # Duplicate the indices of the voxels
        if self.n_repeat > 1:
            vx_idx = np.repeat(vx_idx, self.n_repeat)

        # Return the voxels
        return np.asarray(np.unravel_index(vx_idx, shape), dtype=int).T

    def extract_virtual(self, idx_region, region):
        raise NotImplementedError


class ExtractVoxelRandomly(ExtractVoxel):
    """
    Uniform spatial distribution of the patches
    """
    def __init__(self, n_repeat, n_vx):
        ExtractVoxel.__init__(self, n_repeat)
        self.n_vx = n_vx

    def extract_virtual(self, idx_region, region):
        r = np.random.randint(idx_region.size, size=self.n_vx)
        return idx_region[r]


class ExtractVoxelBalanced(ExtractVoxel):
    """
    Same number of voxels per class
    """
    def __init__(self, n_repeat, n_vx):
        ExtractVoxel.__init__(self, n_repeat)
        self.n_vx = n_vx

    def extract_virtual(self, idx_region, region):
        classes_present = np.unique(region)
        n_classes_present = len(classes_present)
        n_vx_per_class = self.n_vx / n_classes_present
        vx_idx = np.zeros((self.n_vx,), dtype=int)

        for id_k, k in enumerate(classes_present):
            sub_region = np.where(region == k)[0]
            r = np.random.randint(len(sub_region), size=n_vx_per_class)
            vx_idx[id_k * n_vx_per_class:(id_k+1)*n_vx_per_class] = idx_region[sub_region[r]]

        return vx_idx


class ExtractVoxelAll(ExtractVoxel):
    """
    Extract all the possible voxels from the mri region
    """
    def __init__(self, n_repeat):
        ExtractVoxel.__init__(self, n_repeat)

    def extract_virtual(self, idx_region, region):
        return idx_region
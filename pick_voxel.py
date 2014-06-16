__author__ = 'adeb'

import numpy as np

from utilities import distrib_balls_in_bins


def create_pick_voxel(config_ini):
    """
    Factory function to create the objects responsible for picking the voxels
    """
    where_vx = config_ini.pick_vx["where"]
    how_vx = config_ini.pick_vx["how"]
    if where_vx == "anywhere":
        select_region = SelectWholeBrain()
    elif where_vx == "plane":
        axis = config_ini.pick_vx["axis"]
        plane = config_ini.pick_vx["plane"]
        select_region = SelectPlane(axis, plane)
    else:
        print "error in pick_voxel"
        return

    n_patch_per_voxel = config_ini.general["n_patch_per_voxel"]

    if how_vx == "all":
        extract_voxel = ExtractVoxelAll(n_patch_per_voxel)
    else:
        if how_vx == "random":
            extract_voxel = ExtractVoxelRandomly(n_patch_per_voxel)
        elif how_vx == "balanced":
            extract_voxel = ExtractVoxelBalanced(n_patch_per_voxel)
        else:
            print "error in pick_voxel"
            return

    return PickVoxel(select_region, extract_voxel)


class PickVoxel():
    """
    Manage the selection and extraction of voxels in an mri image
    """
    def __init__(self, select_region, extract_voxel):
        self.select_region = select_region
        self.extract_voxel = extract_voxel

    def pick(self, n_vx, label):
        # Select the region in which voxels are going to be extracted
        idx_region = self.select_region.select(label)
        region = label.ravel()[idx_region]
        # Once the region is selected, extract the voxels
        return self.extract_voxel.extract(n_vx, idx_region, region, label.shape)


class SelectRegion():
    """
    Select a specific spatial region of the mri image in which voxels will later be extracted
    """
    def __init__(self):
        pass

    def select(self, label):
        raise NotImplementedError


class SelectWholeBrain(SelectRegion):
    """
    Select the whole labelled brain
    """
    def __init__(self):
        SelectRegion.__init__(self)

    def select(self, label):
        return label.ravel().nonzero()[0]


class SelectPlane(SelectRegion):
    """
    Select a specific orthogonal plane defined by an axis (the plane is orthogonal to this axis) and a specific axis
    coordinate.
    """
    def __init__(self, axis, axis_coordinate):
        SelectRegion.__init__(self)
        self.axis = axis
        self.axis_coordinate = axis_coordinate

    def select(self, label):
        plan = np.zeros(label.shape, dtype=float)
        slice_axis = [slice(None)] * 3
        slice_axis[self.axis] = self.axis_coordinate
        plan[slice_axis] = label[slice_axis]
        return plan.ravel().nonzero()[0]


class ExtractVoxel():
    """
    This class extract voxels from a given region of the mri image
    """
    def __init__(self, n_repeat):
        self.n_repeat = n_repeat

    def extract(self, n_vx, idx_region, region, shape):
        vx_idx = self.extract_virtual(n_vx, idx_region, region)

        # Duplicate the indices of the voxels
        if self.n_repeat > 1:
            vx_idx = np.repeat(vx_idx, self.n_repeat)

        # Return the voxels
        return np.asarray(np.unravel_index(vx_idx, shape), dtype=int).T

    def extract_virtual(self, n_vx, idx_region, region):
        raise NotImplementedError


class ExtractVoxelRandomly(ExtractVoxel):
    """
    Uniform spatial distribution of the patches
    """
    def __init__(self, n_repeat):
        ExtractVoxel.__init__(self, n_repeat)

    def extract_virtual(self, n_vx, idx_region, region):
        r = np.random.randint(idx_region.size, size=n_vx)
        return idx_region[r]


class ExtractVoxelBalanced(ExtractVoxel):
    """
    Same number of voxels per class
    """
    def __init__(self, n_repeat):
        ExtractVoxel.__init__(self, n_repeat)

    def extract_virtual(self, n_vx, idx_region, region):
        vx_idx = np.zeros((n_vx,), dtype=int)

        # Compute the number of voxels for each region
        classes_present = np.unique(region)
        n_classes_present = len(classes_present)
        voxels_per_region = distrib_balls_in_bins(n_vx, n_classes_present)

        vx_counter = 0
        for id_k, k in enumerate(classes_present):
            if voxels_per_region[id_k] == 0:
                continue
            sub_region = np.where(region == k)[0]
            r = np.random.randint(len(sub_region), size=voxels_per_region[id_k])
            vx_counter_next = vx_counter + voxels_per_region[id_k]
            vx_idx[vx_counter:vx_counter_next] = idx_region[sub_region[r]]
            vx_counter = vx_counter_next

        return vx_idx


class ExtractVoxelAll(ExtractVoxel):
    """
    Extract all the possible voxels from the mri region
    """
    def __init__(self, n_repeat):
        ExtractVoxel.__init__(self, n_repeat)

    def extract_virtual(self, n_vx, idx_region, region):
        return idx_region


class ExtractVoxelAllBuffer(ExtractVoxel):
    """
    Extract all the possible voxels from the mri region
    """
    def __init__(self, n_repeat):
        ExtractVoxel.__init__(self, n_repeat)
        self.cur_idx = 0

    def reset(self):
        self.cur_idx = 0

    def extract_virtual(self, n_vx, idx_region, region):
        is_entirely_scanned = False
        next_idx = self.cur_idx + n_vx

        if len(idx_region) > next_idx:
            vx_idx = idx_region[self.cur_idx:]
            is_entirely_scanned = True
        else:
            vx_idx = idx_region[self.cur_idx: next_idx]
            self.cur_idx = next_idx

        return is_entirely_scanned, idx_region
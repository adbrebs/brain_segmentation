__author__ = 'adeb'

import numpy as np
import nibabel as nib
from data_brain_parcellation import crop_brain_and_pad

class BoundaryWeight:
    """
    Experiment to sample more points along the boundaries
    avorted because computing all the distances is intractable.
    """
    def __init__(self):
        self.weights = None

    def compute_weights(self, label):
        shape = label.shape
        self.weights = np.zeros(shape)

        counter = 0
        n = label[label>0].size
        for cur_pos, cur_region in np.ndenumerate(label):
            if cur_region == 0:
                continue
            counter += 1
            print "{} / {}".format(counter, n)
            cube = self.find_cube(cur_pos, cur_region, label)
            self.weights[cur_pos] = 1 / self.compute_distance_boundary(cur_pos, cur_region, cube, label)

    @staticmethod
    def find_cube(cur_pos, cur_region, label):
        shape = label.shape

        def find_first(i):
            d_sup = 0
            d_inf = 0

            pos = np.array(cur_pos)
            while True:
                if pos[i] == 0:
                    d_inf = shape[i]
                    break
                d_inf += 1
                pos[i] -= 1
                if label[tuple(pos)] not in [0, cur_region]:
                    break

            pos = np.array(cur_pos)
            while d_sup < d_inf:
                if pos[i] == shape[i]-1:
                    d_sup = shape[i]
                    break
                d_sup += 1
                pos[i] += 1
                if label[tuple(pos)] not in [0, cur_region]:
                    break

            return min(d_sup, d_inf)

        x_dist = find_first(0)
        y_dist = find_first(1)
        z_dist = find_first(2)
        dist = min(x_dist, y_dist, z_dist)

        x_slice = slice(max(0, cur_pos[0] - dist), min(shape[0], cur_pos[0] + dist + 1))
        y_slice = slice(max(0, cur_pos[1] - dist), min(shape[1], cur_pos[1] + dist + 1))
        z_slice = slice(max(0, cur_pos[2] - dist), min(shape[2], cur_pos[2] + dist + 1))

        return x_slice, y_slice, z_slice

    @staticmethod
    def compute_distance_boundary(cur_pos, cur_region, cube, label):
        zone = label[cube]
        zone[zone == cur_region] = 0
        idx_3d = np.transpose(np.nonzero(zone))
        idx_3d[:, 0] += cube[0].start
        idx_3d[:, 1] += cube[1].start
        idx_3d[:, 2] += cube[2].start
        dists = np.sum((idx_3d - cur_pos)**2, axis=1)
        print np.sqrt(np.min(dists))
        return np.sqrt(np.min(dists))


if __name__ == '__main__':

    mri_file = "./datasets/miccai/1/mri/1000_3.nii"
    lab_file = "./datasets/miccai/1/label/1000_3_glm.nii"

    mri = nib.load(mri_file).get_data().squeeze()
    mri = mri.astype(np.float32, copy=False)
    lab = nib.load(lab_file).get_data().squeeze()
    lab = lab.astype(np.int16, copy=False)

    mri, lab = crop_brain_and_pad(mri, lab, 0)

    b = BoundaryWeight()
    b.compute_weights(lab)
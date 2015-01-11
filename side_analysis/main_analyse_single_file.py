__author__ = 'adeb'

import matplotlib
matplotlib.use('Agg')
import nibabel as nib
import numpy as np

from spynet.utils.utilities import analyse_classes

if __name__ == '__main__':
    nib.nifti1.FLOAT32_EPS_3 = -1e-6
    lab_file = "./datasets/miccai/1000_3.nii"
    lab = nib.load(lab_file).get_data().squeeze()
    lab = np.asarray(lab, dtype=np.int16)

    analyse_classes(lab[lab.nonzero()])


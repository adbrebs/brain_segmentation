__author__ = 'adeb'

import matplotlib
matplotlib.use('Agg')

from spynet.utils.utilities import compare_two_seg

if __name__ == '__main__':

    true_seg_path = "./true_seg.nii"
    pred_seg_path = "./pred_seg.nii"
    compare_two_seg(pred_seg_path, true_seg_path)


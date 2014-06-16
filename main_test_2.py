__author__ = 'adeb'

import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import nibabel

from utilities import compare_two_seg
from network import load_network
import trainer
from pick_patch import *
from pick_target import *
from data_generator import list_miccai_files, crop_image


if __name__ == '__main__':

    true_seg_path = "./true_seg.nii"
    pred_seg_path = "./pred_seg.nii"
    compare_two_seg(pred_seg_path, true_seg_path)


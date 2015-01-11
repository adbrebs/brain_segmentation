__author__ = 'adeb'

import os
import imp
import time
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import nibabel
import theano
import theano.tensor as T
import scipy.io

from spynet.utils.utilities import create_img_from_pred, compute_dice_symb, compute_dice, error_rate
import spynet.training.trainer as trainer
from spynet.models.network import *
from network_brain_parcellation import *
from spynet.data.utils_3d.pick_voxel import *
from spynet.data.utils_3d.pick_patch import *
from spynet.data.utils_3d.pick_target import *
from data_brain_parcellation import DatasetBrainParcellation, DataGeneratorBrain, list_miccai_files, RegionCentroids
from spynet.utils.utilities import open_h5file


if __name__ == '__main__':
    """
    Evaluate a trained network (without approximating the centroids)
    """

    experiment_path = "./experiments/paper_ultimate_conv/"
    data_path = "./datasets/paper_ultimate_conv/"

    # Load the network
    net = NetworkUltimateConv()
    net.init(33, 29, 5, 134, 135)
    net.load_parameters(open_h5file(experiment_path + "net.net"))
    n_out = net.n_out

    # Load the scaler
    scaler = pickle.load(open(experiment_path + "s.scaler", "rb"))

    testing_data_path = data_path + "test.h5"
    ds_testing = DatasetBrainParcellation()
    ds_testing.read(testing_data_path)
    scaler.scale(ds_testing.inputs)

    out_pred = net.predict(ds_testing.inputs, 1000)
    errors = np.argmax(out_pred, axis=1) != np.argmax(ds_testing.outputs, axis=1)
    dice = compute_dice(np.argmax(out_pred, axis=1), np.argmax(ds_testing.outputs, axis=1), 134)
    print np.mean(dice)
    print np.mean(errors)
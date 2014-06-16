__author__ = 'adeb'

import os
import sys
import h5py
import numpy as np
import random

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import theano
import nibabel as nib


def load_config(default_file):
    """
    Load a config file specified in the command line.
    Attributes:
        default_file (string): If no *args is provided, use this argument.
    """
    if len(sys.argv) == 1:
        cf = __import__(default_file)
    else:
        cf = __import__(str(sys.argv[1]))

    return cf


def create_directories(folder_name):
    # Create directories if they don't exist
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    experiments_dir = "./experiments"
    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)
    exp_dir = experiments_dir + "/" + folder_name + "/"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    return exp_dir


def share(data, name=None, borrow=True):
    """
    Transform data into Theano shared variable.
    """
    return theano.shared(np.asarray(data, dtype=theano.config.floatX), name=name, borrow=borrow)


def get_h5file_attribute(h5file, attr_key):
    try:
        attr_value = h5file.attrs[attr_key]
    except KeyError:
        raise Exception("Attribute {} is not present in {}".format(attr_key, h5file.filename))

    return attr_value


def get_h5file_data(h5file, data_key):
    try:
        data_value = h5file[data_key].value
    except KeyError:
        raise Exception("Data {} is not present in {}".format(data_key, h5file.filename))

    return data_value


def open_h5file(file_path):
    try:
        h5file = h5py.File(file_path, "r")
    except IOError:
        raise Exception("{} does not exist".format(file_path))

    return h5file


def distrib_balls_in_bins(n_balls, n_bins):
    """
    Uniformly distribute n_balls in n_bins
    """
    balls_per_bin = np.zeros((n_bins,), dtype=int)
    div, n_balls_remaining = divmod(n_balls, n_bins)
    balls_per_bin += div
    rand_idx = np.asarray(random.sample(range(n_bins), n_balls_remaining), dtype=int)
    balls_per_bin[rand_idx] += 1
    return balls_per_bin


def create_img_from_pred(vx, pred, shape):
    """
    Create a labelled image array from voxels and their predictions
    """
    pred_img = np.zeros(shape, dtype=np.uint8)
    if len(shape) == 2:
        pred_img[vx[:, 0], vx[:, 1]] = pred
    elif len(shape) == 3:
        pred_img[vx[:, 0], vx[:, 1], vx[:, 2]] = pred
    return pred_img


def analyse_targets(targets_scalar, verbose=True):
    """
    Compute various statistics about targets.
    """

    targets_scalar = targets_scalar[targets_scalar.nonzero()]

    a = np.bincount(targets_scalar)
    classes = np.nonzero(a)[0]
    n_classes = len(classes)
    if verbose:
        print("There are {} regions in the dataset".format(n_classes))
    proportion_volumes = a[classes].astype(float, copy=False)
    proportion_volumes /= sum(proportion_volumes)

    if verbose:
        print("    The largest region represents {} % of the image".format(max(proportion_volumes) * 100))
        print("    The smallest region represents {} % of the image".format(min(proportion_volumes) * 100))

    return classes, proportion_volumes


def compute_dice(img_pred, img_true, n_classes_max):
    """
    Compute the DICE score between two segmentations
    """
    classes = np.unique(img_pred)
    if classes[0] == 0:
        classes = classes[1:]
    dices = np.zeros((n_classes_max,))

    for c in classes:
        class_pred = img_pred == c
        class_true = img_true == c
        class_common = class_true[class_pred]
        dices[c] = 2 * np.sum(np.asarray(class_common, dtype=float)) / (np.sum(class_pred) + np.sum(class_true))

    return dices


def compare_two_seg(pred_seg_path, true_seg_path):
    pred_seg = nib.load(pred_seg_path).get_data().squeeze()
    true_seg = nib.load(true_seg_path).get_data().squeeze()

    classes, true_volumes = analyse_targets(np.ravel(true_seg))
    dices = compute_dice(pred_seg, true_seg, len(classes)+1)
    dices = dices[1:]

    # Plot dice in function of log volume
    plt.plot(np.log10(true_volumes), dices, 'ro', label="one region")
    plt.xlabel('Log-volume of the region')
    plt.ylabel('Dice coefficient of the region')
    plt.savefig("./analysis/log_volume.png")

    # Plot dice in function of the sorted indices of the regions
    plt.figure()
    idx = np.argsort(dices)
    plt.plot(idx, dices[idx], 'ro', label="one region")
    plt.xlabel('Sorted indices of the regions (the higher the bigger the region)')
    plt.ylabel('Dice coefficient of the sorted region')
    plt.savefig("./analysis/dices_sorted.png")
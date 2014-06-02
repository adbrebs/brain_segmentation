__author__ = 'adeb'

import os
import sys
import h5py
import numpy as np

import theano


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


def analyse_data(targets):
    """
    Compute various statistics about targets.
    """

    # Number of classes
    targets_scalar = np.argmax(targets, axis=1)
    classes = np.unique(targets_scalar)
    n_classes = len(classes)
    print("There are {} regions in the dataset".format(n_classes))

    a = np.bincount(targets_scalar)
    b = np.nonzero(a)[0]
    c = a[b].astype(float, copy=False)
    c /= sum(c)

    print("    The largest region represents {} % of the image".format(max(c) * 100))

    return b, c


def compute_dice(img_pred, img_true, n_classes_max):
    """
    Compute the DICE score between two segmentations
    """
    classes = np.unique(img_pred)
    if classes[0] == 0:
        classes = classes[1:]
    dices = np.zeros((n_classes_max, 1))

    for c in classes:
        class_pred = img_pred == c
        class_true = img_true == c
        class_common = class_true[class_pred]
        dices[c] = 2 * np.sum(np.asarray(class_common, dtype=float)) / (np.sum(class_pred) + np.sum(class_true))

    return dices
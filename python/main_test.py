__author__ = 'adeb'

import sys
import ConfigParser

from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
import theano
import theano.sandbox.cuda

import nn
import trainer
from data_generation import ConverterMriPatch, create_img_from_pred
from pick_voxel import *
from pick_patch import *
from pick_target import *


def load_config():
    cf = ConfigParser.ConfigParser()
    if len(sys.argv) == 1:
        cf.read('training.ini')
    else:
        cf.read(str(sys.argv[1]))
    theano.sandbox.cuda.use(cf.get('general', 'gpu'))
    return cf

if __name__ == '__main__':

    patch_width = 29
    n_classes = 139

    ### Load the network
    net = nn.Network2(patch_width, n_classes)
    net.load_parameters("net2.net")

    ### Create the patches
    mri_file = '../data/miccai/mri/1000.nii'
    label_file = '../data/miccai/label/1000.nii'
    select_region = SelectPlaneXZ(100)
    extract_voxel = ExtractVoxelRandomly(1,500)
    pick_vx = PickVoxel(select_region, extract_voxel)
    pick_patch = PickPatchParallelOrthogonal(1)
    pick_tg = PickTgCentered()
    conv_mri_patch = ConverterMriPatch(patch_width, pick_vx, pick_patch, pick_tg)
    vx, patch, idx_patch, tg, mri, lab = conv_mri_patch.convert(mri_file, label_file, n_classes)

    slice1 = mri[:,:,100]
    mri_rot = rotate(mri, 20, axes=(0,1))
    slice2 = mri_rot[:,:,100]

    # plt.imshow(slice1)
    # plt.show()
    #
    # plt.imshow(slice2)
    # plt.show()

    ### Predict the patches
    pred = net.predict(patch)
    pred2 = np.argmax(pred, axis=1)
    err = trainer.Trainer.error_rate(pred, tg)
    print err
    true = lab[:, 100, :]
    img_pred = create_img_from_pred(vx[:, (0, 2)], pred2, true.shape)

    file_name = "test.png"
    plt.imshow(img_pred)
    plt.savefig('../images/pred_' + file_name)

    plt.imshow(true)
    plt.savefig('../images/true_' + file_name)

    plt.imshow(img_pred != true)
    plt.savefig('../images/diff_' + file_name)

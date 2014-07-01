__author__ = 'adeb'

import os
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import nibabel
import theano
import theano.tensor as T

from spynet.utils.utilities import create_img_from_pred, compute_dice_symb
import spynet.training.trainer as trainer
from network_brain_parcellation import NetworkUltimate
from spynet.data.utils_3d.pick_patch import *
from spynet.data.utils_3d.pick_target import *
from data_brain_parcellation import list_miccai_files, crop_image
from spynet.utils.utilities import open_h5file


def evaluate_network_on_dataset(net, pick_patch, pick_tg, n_classes, compute_dice_c):
    file_list = list_miccai_files(**{"mode": "idx_files", "idx_files": [16]})
    dice_regions = 0
    for file in file_list[0:1]:
        dice_regions += evaluate_network_on_brain(net, file[0], file[1], pick_patch, pick_tg, n_classes, compute_dice_c)


def evaluate_network_on_brain(net, mri_file, label_file, pick_patch, pick_tg, n_classes, compute_dice_c):

    print "Evaluating file {}".format(mri_file)

    start_time = time.clock()

    # Load the mri data
    mri = nibabel.load(mri_file).get_data().squeeze()
    lab = nibabel.load(label_file).get_data().squeeze()
    mri, lab = crop_image(mri, lab)
    lab = np.asarray(lab, dtype=np.int8)
    affine = nibabel.load(label_file).get_affine()

    is_scan_finished = False
    ls_vx = []
    ls_pred = []
    batch_size = 100000
    pred_function = net.generate_testing_function(batch_size)
    idx_brain = lab.ravel().nonzero()[0]
    n_vx = len(idx_brain)
    cur_idx = 0

    while not is_scan_finished:
        next_idx = cur_idx + batch_size
        print "\r     voxels [{} - {}] / {}".format(cur_idx, next_idx, n_vx)

        if next_idx >= len(idx_brain):
            vx_idx = idx_brain[cur_idx:]
            is_scan_finished = True
            pred_function = net.generate_testing_function(len(idx_brain) - cur_idx)
        else:
            vx_idx = idx_brain[cur_idx: next_idx]
            cur_idx = next_idx

        start = time.time()
        vx = np.asarray(np.unravel_index(vx_idx, mri.shape), dtype=int).T
        patch, idx_patch = pick_patch.pick(vx, mri, lab)
        tg = pick_tg.pick(vx, idx_patch, n_classes, mri, lab)

        net.scale_raw_data(patch)

        ### Predict the patches
        pred_raw = pred_function(patch)
        pred = np.argmax(pred_raw, axis=1)
        err = trainer.Trainer.error_rate(pred_raw, tg)
        print err
        ls_vx.append(vx)
        ls_pred.append(pred)

    # Count the number of voxels
    n_vx = 0
    for vx in ls_vx:
        n_vx += vx.shape[0]

    # Aggregate the data
    vx_all = np.zeros((n_vx, 3), dtype=int)
    pred_all = np.zeros((n_vx,), dtype=int)
    idx = 0
    for vx, pred in zip(ls_vx, ls_pred):
        next_idx = idx+vx.shape[0]
        vx_all[idx:next_idx] = vx
        pred_all[idx:next_idx] = pred
        idx = next_idx

    img_true = lab
    img_pred = create_img_from_pred(vx_all, pred_all, img_true.shape)

    dice_regions = compute_dice_c(img_pred.ravel(), img_true.ravel())

    ### Save the brain images
    # file_name = "test.png"
    # plt.imshow(img_pred)
    # plt.savefig('./images/pred_' + file_name)
    #
    # plt.imshow(img_true)
    # plt.savefig('./images/true_' + file_name)
    #
    # plt.imshow(img_pred != img_true)
    # plt.savefig('./images/diff_' + file_name)

    end_time = time.clock()
    print "It took {} seconds to evaluate the whole brain.".format(end_time - start_time)
    print "The mean dice is" + str(dice_regions.mean())

    basename = os.path.splitext(os.path.basename(mri_file))[0]
    img_pred_nifti = nibabel.Nifti1Image(img_pred, affine)
    img_mri_nifti = nibabel.Nifti1Image(mri, affine)
    img_true_nifti = nibabel.Nifti1Image(img_true, affine)
    data_path = "./datasets/seg_res/"
    nibabel.save(img_pred_nifti, data_path + basename + "_pred.nii")
    nibabel.save(img_mri_nifti, data_path + basename + "_mri.nii")
    nibabel.save(img_true_nifti, data_path + basename + "_true.nii")

    return dice_regions


if __name__ == '__main__':

    patch_width = 29
    n_classes = 139

    ### Load the network
    net = NetworkUltimate()
    net.load_parameters(open_h5file("./experiments/miccai_best/net.net"))

    # Options for the dataset
    pick_patch = PickUltimate(patch_width)  # PickPatchParallelOrthogonal(patch_width, 1)
    pick_tg = PickTgCentered()

    # Compile DICE function
    pred = T.vector("pred")
    true = T.vector("true")

    compute_dice_c = theano.function(
        inputs=[pred, true],
        outputs=compute_dice_symb(pred, true, n_classes))

    evaluate_network_on_dataset(net, pick_patch, pick_tg, n_classes, compute_dice_c)


__author__ = 'adeb'

from utilities import load_config
from dataset import DatasetBrainParcellation
from data_generator import DataGeneratorBrain, list_miccai_files
from pick_voxel import *
from pick_patch import *
from pick_target import *


def create_dataset(file_path, patch_width, n_data, miccai_folder):
    # Create the data generator
    select_region = SelectWholeBrain()
    extract_voxel = ExtractVoxelRandomly(1)
    pick_vx = PickVoxel(select_region, extract_voxel)
    pick_patch = PickUltimate(patch_width)
    pick_tg = PickTgCentered()

    files = list_miccai_files(miccai_folder)
    dg = DataGeneratorBrain()
    dg.init_from(files, pick_vx, pick_patch, pick_tg, 1)
    vx, patch, idx_patch, tg, file_id = dg.generate(n_data)

    dc_training = DatasetBrainParcellation()
    dc_training.populate(patch, tg, vx, idx_patch, file_id, patch_width)
    dc_training.write(file_path)


if __name__ == '__main__':

    patch_width = 29

    file_path_train = "./data/ultimate_train.h5"
    n_data_train = 100000
    miccai_folder_train = "1"
    create_dataset(file_path_train, patch_width, n_data_train, miccai_folder_train)

    file_path_test = "./data/ultimate_test.h5"
    n_data_test = 10000
    miccai_folder_test = "2"
    create_dataset(file_path_train, patch_width, n_data_train, miccai_folder_train)
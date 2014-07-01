__author__ = 'adeb'

import os
from shutil import copy2
import inspect


data_path = "./datasets/test/"


general = {
    "source": "miccai",
    "source_kwargs": {"mode": "miccai_challenge", "source_folder": "2"},
    "n_patch_per_voxel": 1,
    "file_path": data_path + "test.h5",
    "perm": False,
    "n_data": 10000
}

pick_vx = {
    "where": "anywhere",  # anywhere, plane
    "plane": 100,
    "axis": 1,
    "how": "balanced"  # all, random, balanced
}

pick_patch = {
    "how": "2Dortho",  # 2Dortho, 2DorthoRotated, 3D, ultimate
    "patch_width": 29,
    "axis": 1,
    "max_degree_rotation": 30
}

pick_tg = {
    "how": "center"  # center, proportion
}


############################################################
############# this part should not be modified #############
############################################################

# Create the folder if it does not exist
if not os.path.exists(data_path):
    os.makedirs(data_path)

# Copy the config file
copy2(inspect.getfile(inspect.currentframe()), data_path)

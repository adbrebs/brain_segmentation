__author__ = 'adeb'

data_path = "./datasets/final_3_compressed_patches_random/"

general = {
    "source": "miccai",
    "source_kwargs": {"mode": "folder", "path": "./datasets/miccai/1/"},
    "file_path": data_path + "test.h5",
    "perm": False,
    "n_data": 40000
}

pick_vx = {
    "where": "anywhere",  # anywhere, plane
    "plane": 100,
    "axis": 1,
    "how": "random"  # all, random, balanced
}

pick_features = [
    # types: 2Dortho, 3D, centroid
    {
        "how": "2Dortho",  # 2Dortho, 2DorthoRotated, 3D, three_patches, local_global, grid_patches
        "patch_width": 29,
        "axis": [0, 1, 2],
        "scale": 3
    },
    # {
    #     "how": "2Dortho",  # 2Dortho, 2DorthoRotated, 3D, three_patches, local_global, grid_patches
    #     "patch_width": 29,
    #     "axis": [0, 1, 2],
    #     "scale": 3
    # },
    # {
    #     "how": "3D",  # 2Dortho, 2DorthoRotated, 3D, three_patches, local_global, grid_patches
    #     "patch_width": 13,
    #     "scale": 1
    # },
    # {
    #     "how": "centroid",  # 2Dortho, 2DorthoRotated, 3D, three_patches, local_global, grid_patches
    #     "n_features": 134
    # }
]

pick_tg = {
    "how": "center"  # center, proportion
}
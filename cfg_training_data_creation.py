__author__ = 'adeb'


general = {
    "source": "miccai",
    "source_folder": "1",
    "n_patch_per_voxel": 1,
    "file_path": None,
    "perm": True,
    "n_data": 500000
}

pick_vx = {
    "where": "anywhere",  # anywhere, plane
    "plane": 100,
    "axis": 1,
    "how": "balanced"  # all, random, balanced
}

pick_patch = {
    "how": "ultimate",  # 2Dortho, 2DorthoRotated, 3D, ultimate
    "patch_width": 29,
    "axis": 1,
    "max_degree_rotation": 30
}

pick_tg = {
    "how": "center"  # center, proportion
}
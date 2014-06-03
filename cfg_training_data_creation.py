__author__ = 'adeb'


class CF:
    def __init__(self, file_path):

        self.general = {
            "source": "miccai",
            "source_folder": "1",
            "n_patch_per_voxel": 1,
            "file_path": file_path,
            "perm": True,
            "n_data": 10000
        }

        self.pick_vx = {
            "where": "anywhere",  # anywhere, plane
            "plane": 100,
            "axis": 1,
            "how": "random"  # all, random, balanced
        }

        self.pick_patch = {
            "how": "2Dortho",  # 2Dortho, 2DorthoRotated, 3D
            "patch_width": 29,
            "axis": 1,
            "max_degree_rotation": 20
        }

        self.pick_tg = {
            "how": "center"  # center, proportion
        }
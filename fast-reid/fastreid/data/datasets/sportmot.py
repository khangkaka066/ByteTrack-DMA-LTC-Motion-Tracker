# encoding: utf-8

import glob
import os.path as osp
import re

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class SportMOT(ImageDataset):
    """SportMOT ReID patches generated from MOT-format annotations."""

    dataset_dir = "SportMOT-ReID"
    dataset_name = "sportmot"

    def __init__(self, root="datasets", **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super(SportMOT, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, "*.jpg"))
        pattern = re.compile(r"([-\d]+)_c(\d+)")

        data = []
        for img_path in img_paths:
            match = pattern.search(osp.basename(img_path))
            if match is None:
                continue
            pid, camid = map(int, match.groups())
            if pid <= 0:
                continue
            camid -= 1
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data

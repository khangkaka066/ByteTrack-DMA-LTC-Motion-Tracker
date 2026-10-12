# encoding: utf-8
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "mot"))
from yolox_nano_mix_det import Exp as NanoExp  # noqa: E402

class OfficialValTransform:
    """Eval transform of upstream Megvii YOLOX (>=0.2): BGR, [0, 255], no normalisation."""

    def __call__(self, img, res, input_size):
        from yolox.data.data_augment import preproc

        img, _ = preproc(img, input_size, None, None)
        # preproc swaps BGR->RGB and divides by 255; undo both.
        img = np.ascontiguousarray(img[::-1] * 255.0, dtype=np.float32)
        return img, np.zeros((1, 5))


ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", ".."))


class Exp(NanoExp):
    """YOLOX-nano evaluated on MuseumVisitors (built by tools/prepare_museum_mot.py).

    Override from the command line, e.g. `test_h 416 test_w 640 preproc legacy`
    (tuples/bools cannot be passed safely through exp.merge).

    preproc=legacy   -> ByteTrack ValTransform (RGB, /255, ImageNet mean/std)
    preproc=official -> upstream YOLOX ValTransform (BGR, 0..255), which is what
                        pretrained/yolox-nano.pth (trained with Megvii YOLOX) expects.
    Ultralytics detectors convert either layout back to RGB [0, 1] internally.
    """

    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_dir = os.path.join(ROOT, "datasets", "museum_mot")
        self.val_ann = "test.json"
        self.val_name = "test"
        # 1280x800 frames scaled by 0.5 -> 640x400, padded to a multiple of 32.
        # This matches the Ultralytics fine-tuning scale (imgsz=640).
        self.test_h = 416
        self.test_w = 640
        self.test_size = (self.test_h, self.test_w)
        self.preproc = "official"
        self.data_num_workers = 4

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform

        self.test_size = (int(self.test_h), int(self.test_w))
        assert self.preproc in ("legacy", "official"), self.preproc
        valdataset = MOTDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            img_size=self.test_size,
            name=self.val_name,
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ) if self.preproc == "legacy" else OfficialValTransform(),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        return torch.utils.data.DataLoader(
            valdataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.data_num_workers,
            pin_memory=True,
        )

    @property
    def legacy_preproc(self):
        return self.preproc == "legacy"

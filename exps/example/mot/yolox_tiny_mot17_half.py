# encoding: utf-8
"""YOLOX-tiny trained on MOT17 train_half only, evaluated on val_half.

Hyper-parameters follow Table 2 of "Performance Comparison of YOLOv10, YOLOv11,
and YOLOv12 Models on Human Detection Datasets" (Brilliance, 2025) so YOLOX-tiny
can be compared with YOLOv10n / YOLO11n / YOLO12n trained the same way:
SGD, lr0 0.01, final lr ratio 0.1, momentum 0.937, batch 16, 100 epochs,
10 warmup epochs, mosaic prob 0.5, mosaic closed for the last 20 epochs, cosine LR.
Input size keeps the ByteTrack MOT default for YOLOX-tiny (608x1088).
Start from the COCO-pretrained yolox_tiny.pth, not a ByteTrack MOT checkpoint.
"""
import os
import random

import torch
import torch.distributed as dist

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.375
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.train_ann = "train_half.json"
        self.val_ann = "val_half.json"
        self.input_size = (608, 1088)
        self.test_size = (608, 1088)
        # Ultralytics trains at a fixed size by default, so no multi-scale here.
        self.random_size = None

        # Table 2 of the paper
        self.max_epoch = 100
        self.warmup_epochs = 10
        self.no_aug_epochs = 20          # close_mosaic = 20
        self.basic_lr_per_img = 0.01 / 16.0  # lr0 = 0.01 at batch 16
        self.min_lr_ratio = 0.1          # lrf = 0.1
        self.momentum = 0.937
        self.weight_decay = 5e-4
        self.scheduler = "yoloxwarmcos"  # cosine LR
        self.mosaic_prob = 0.5

        # Augmentations matched to Ultralytics defaults (no rotation/shear/mixup)
        self.degrees = 0.0
        self.translate = 0.1
        self.scale = (0.5, 1.5)
        self.shear = 0.0
        self.perspective = 0.0
        self.enable_mixup = False

        self.data_num_workers = 4
        self.print_interval = 20
        self.eval_interval = 1
        self.test_conf = 0.001
        self.nmsthre = 0.7

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )
        mosaic_prob = self.mosaic_prob

        class ProbMosaicDetection(MosaicDetection):
            """MosaicDetection that only builds a mosaic with probability `mosaic_prob`.

            The batch sampler passes (input_dim, idx, mosaic_flag); the parent's
            resize_getitem wrapper reads the flag per call, so we only flip it here.
            """

            def __getitem__(self, index):
                if isinstance(index, (tuple, list)) and index[2] and random.random() >= mosaic_prob:
                    index = (index[0], index[1], False)
                return MosaicDetection.__getitem__(self, index)

        dataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "mot"),
            json_file=self.train_ann,
            name="train",
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )

        dataset = ProbMosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        return DataLoader(self.dataset, **dataloader_kwargs)

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform

        valdataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "mot"),
            json_file=self.val_ann,
            img_size=self.test_size,
            name="train",
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
            "batch_size": batch_size,
        }
        return torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        return COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )

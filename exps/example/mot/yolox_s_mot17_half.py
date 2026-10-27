# encoding: utf-8
"""YOLOX-s trained on MOT17 train_half only, evaluated on val_half.

Same data split, hyper-parameters (Table 2 of the Brilliance 2025 paper) and
augmentations as yolox_tiny_mot17_half.py; only the network size changes.
YOLOX-s (~9.0M params) matches the Ultralytics "s" models (7-9.5M params).
Input size keeps the ByteTrack MOT default (608x1088).
Start from the COCO-pretrained yolox_s.pth, not a ByteTrack MOT checkpoint.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from yolox_tiny_mot17_half import Exp as TinyHalfExp  # noqa: E402


class Exp(TinyHalfExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

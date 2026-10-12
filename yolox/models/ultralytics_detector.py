"""Wrap an Ultralytics YOLO checkpoint so MOTEvaluator can use it like YOLOX.

The evaluator feeds batches produced by the exp's eval transform (letterboxed to
``test_size`` with top-left padding). This module converts them to the RGB
[0, 1] layout Ultralytics expects, runs the raw network + Ultralytics NMS, and
returns the post-processed YOLOX format ``[x1, y1, x2, y2, obj, cls_conf, cls]``
in input-image coordinates. Because NMS already happens here, the evaluator
must skip ``yolox.utils.postprocess`` (see ``is_postprocessed``).
"""
import torch
import torch.nn as nn

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class UltralyticsDetector(nn.Module):
    is_postprocessed = True

    def __init__(self, weights, conf=0.01, iou=0.7, max_det=300, legacy_preproc=True, person_class=None):
        super().__init__()
        from ultralytics import YOLO

        yolo = YOLO(weights)
        self.names = yolo.names
        self.model = yolo.model.fuse(verbose=False).eval()
        self.end2end = bool(getattr(self.model, "end2end", False))
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.legacy_preproc = legacy_preproc
        if person_class is None:
            ids = [k for k, v in self.names.items() if v.lower() in ("person", "pedestrian")]
            person_class = ids[0] if ids else 0
        self.classes = [person_class] if len(self.names) > 1 else None
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1), persistent=False)

    def _to_rgb01(self, imgs):
        if self.legacy_preproc:
            # ByteTrack ValTransform: RGB, /255, ImageNet mean/std normalised.
            return imgs * self.std.to(imgs.dtype) + self.mean.to(imgs.dtype)
        # Official YOLOX ValTransform: BGR in [0, 255].
        return imgs.flip(1) / 255.0

    @torch.no_grad()
    def forward(self, imgs):
        from ultralytics.utils import nms

        x = self._to_rgb01(imgs).clamp_(0, 1)
        preds = self.model(x)
        dets = nms.non_max_suppression(
            preds, self.conf, self.iou, self.classes,
            max_det=self.max_det, end2end=self.end2end,
        )
        outputs = []
        for d in dets:
            if d is None or d.shape[0] == 0:
                outputs.append(None)
                continue
            d = d.float()
            ones = torch.ones_like(d[:, 4:5])
            outputs.append(torch.cat([d[:, :4], d[:, 4:5], ones, torch.zeros_like(ones)], dim=1))
        return outputs

import cv2
import numpy as np
import torch
import torch.nn.functional as F
# from torch.backends import cudnn

from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch

# NOTE: tried cudnn.benchmark = True here (see optimize.txt) — helps in an
# isolated benchmark with a FIXED detection count, but real MOT footage has a
# different detection count almost every frame, so the last torch.split
# chunk's shape keeps changing; cudnn.benchmark then pays for a fresh
# per-shape autotune search often enough to regress the full pipeline
# (~+4ms track time, measured). Left disabled.
# cudnn.benchmark = True


def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False

    cfg.freeze()

    return cfg


def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features


def preprocess(image, input_size):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[1], input_size[0], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size) * 114
    img = np.array(image)
    r = min(input_size[1] / img.shape[0], input_size[0] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    return padded_img, r


class FastReIDInterface:
    def __init__(self, config_file, weights_path, device, batch_size=16):
        super(FastReIDInterface, self).__init__()
        if device != 'cpu':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.batch_size = batch_size    # 8

        self.cfg = setup_cfg(config_file, ['MODEL.WEIGHTS', weights_path])

        self.model = build_model(self.cfg)
        self.model.eval()

        Checkpointer(self.model).load(weights_path)

        if self.device != 'cpu':
            self.model = self.model.eval().to(device='cuda').half()
        else:
            self.model = self.model.eval()

        self.pH, self.pW = self.cfg.INPUT.SIZE_TEST     # [384, 128]

    def inference(self, image, detections):

        if detections is None or np.size(detections) == 0:
            return []

        H, W, _ = np.shape(image)       # original size, [1080, 1920] for MOT17

        patches = []
        for d in range(np.size(detections, 0)):     # iteration over detections
            tlbr = detections[d, :4].astype(np.float32)
            x1 = max(0, min(W - 1, int(np.floor(tlbr[0]))))
            y1 = max(0, min(H - 1, int(np.floor(tlbr[1]))))
            x2 = max(0, min(W, int(np.ceil(tlbr[2]))))
            y2 = max(0, min(H, int(np.ceil(tlbr[3]))))
            if x2 <= x1 or y2 <= y1:
                continue
            patch = image[y1:y2, x1:x2, :]      # crop image, BGR

            # the model expects RGB inputs
            patch = patch[:, :, ::-1]

            # Apply pre-processing to image.
            patch = cv2.resize(patch, tuple(self.cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_LINEAR)    # [384, 128, 3]
            patches.append(patch)

        if not patches:
            return np.zeros((0, 2048))          # TODO: [hgx1001] need to be set by hand

        # Stack every crop for this frame into one uint8 array and move it to
        # the device in a single transfer, instead of one .to(device) call per
        # crop. uint8 is also 4x smaller than the float32 the old code sent
        # over PCIe, so less data moves for the same crops.
        batch_np = np.stack(patches, axis=0)              # (N, H, W, 3) uint8, RGB
        batch = torch.from_numpy(batch_np).to(device=self.device, non_blocking=True)
        batch = batch.permute(0, 3, 1, 2).contiguous()    # (N, 3, H, W)
        batch = batch.half() if self.device != 'cpu' else batch.float()

        # Keep compute chunked at self.batch_size (bounds GPU memory for the
        # conv backbone), but only cross the GPU->CPU boundary once per frame:
        # postprocess()'s .cpu() call forces a full device sync, so calling it
        # per-chunk paid for that sync n_chunks times instead of once.
        preds = []
        with torch.no_grad():
            for chunk in torch.split(batch, self.batch_size, dim=0):
                pred = self.model(chunk)          # [B, 2048]
                pred[torch.isinf(pred)] = 1.0
                preds.append(pred)

        return postprocess(torch.cat(preds, dim=0))   # normalization() and numpy()

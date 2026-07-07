"""
Generate det/det.txt for every sequence under a dataset split.

Runs YOLOX detector (no tracking) on each frame and saves raw detections
in MOT format: frame_id,-1,x,y,w,h,conf,-1,-1,-1

Usage:
    python3 tools/gen_detections.py \
        -f exps/example/dancetrack/yolox_x_dancetrack.py \
        -c pretrained/bytetrack_dancetrack.pth.tar \
        --split-dir datasets/dancetrack/train \
        --conf 0.01 \
        --device gpu
"""

import argparse
import os
import sys

FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(FILE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2
import torch
from loguru import logger
from tqdm import tqdm

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess


# ---------------------------------------------------------------------------
# Predictor (detection only, no tracking)
# ---------------------------------------------------------------------------

class Predictor:
    def __init__(self, model, exp, device, fp16=False):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    @torch.no_grad()
    def infer(self, img_path: str):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"cv2.imread returned None for: {img_path!r}")
        h, w = img.shape[:2]
        img_t, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_t = torch.from_numpy(img_t).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img_t = img_t.half()
        outputs = self.model(img_t)
        outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs[0], ratio, h, w


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_sequences(split_dir: str) -> list[str]:
    """Return paths to every subdirectory that contains img1/."""
    seqs = []
    for name in sorted(os.listdir(split_dir)):
        seq = os.path.join(split_dir, name)
        if os.path.isdir(os.path.join(seq, "img1")):
            seqs.append(seq)
    return seqs


def _sorted_frames(seq_dir: str) -> list[str]:
    img_dir = os.path.abspath(os.path.join(seq_dir, "img1"))
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    frames = sorted(
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in exts and not f.startswith("._")
    )
    return frames


def _run_sequence(seq_dir: str, predictor: Predictor, conf_thresh: float, overwrite: bool):
    det_dir = os.path.join(seq_dir, "det")
    det_path = os.path.join(det_dir, "det.txt")

    if os.path.exists(det_path) and not overwrite:
        logger.info(f"  skip (already exists): {det_path}")
        return

    os.makedirs(det_dir, exist_ok=True)
    frames = _sorted_frames(seq_dir)

    lines = []
    for frame_id, img_path in enumerate(frames, 1):
        output, ratio, h, w = predictor.infer(img_path)
        if output is None:
            continue
        output = output.cpu()
        # columns: x1, y1, x2, y2, obj_conf, cls_conf, cls_id
        bboxes = output[:, :4] / ratio
        scores = (output[:, 4] * output[:, 5]).numpy()
        bboxes = bboxes.numpy()

        for (x1, y1, x2, y2), s in zip(bboxes, scores):
            if s < conf_thresh:
                continue
            x1 = max(0.0, x1)
            y1 = max(0.0, y1)
            x2 = min(float(w), x2)
            y2 = min(float(h), y2)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            lines.append(
                f"{frame_id},-1,{x1:.2f},{y1:.2f},{bw:.2f},{bh:.2f},{s:.4f},-1,-1,-1\n"
            )

    with open(det_path, "w") as f:
        f.writelines(lines)

    logger.info(f"  saved {len(lines)} detections → {det_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def make_parser():
    p = argparse.ArgumentParser("Generate det/det.txt for each sequence")
    p.add_argument("-f", "--exp_file", required=True, help="YOLOX exp file")
    p.add_argument("-c", "--ckpt", required=True, help="checkpoint path")
    p.add_argument("--split-dir", required=True,
                   help="dataset split directory (e.g. datasets/dancetrack/train)")
    p.add_argument("--conf", type=float, default=0.01,
                   help="minimum detection confidence to save (default 0.01, keep low for pseudo-labels)")
    p.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--fuse", action="store_true", help="fuse conv+bn for speed")
    p.add_argument("--overwrite", action="store_true",
                   help="re-generate even if det.txt already exists")
    p.add_argument("-n", "--name", default=None)
    p.add_argument("--tsize", type=int, default=None)
    return p


def main():
    args = make_parser().parse_args()

    exp = get_exp(args.exp_file, args.name)
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    device = torch.device("cuda" if args.device == "gpu" else "cpu")

    model = exp.get_model().to(device)
    model.eval()

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    logger.info(f"Loaded checkpoint: {args.ckpt}")

    if args.fuse:
        model = fuse_model(model)
    if args.fp16:
        model = model.half()

    predictor = Predictor(model, exp, device, fp16=args.fp16)

    seqs = _find_sequences(args.split_dir)
    logger.info(f"Found {len(seqs)} sequences under {args.split_dir}")

    for seq_dir in tqdm(seqs, unit="seq"):
        seq_name = os.path.basename(seq_dir)
        tqdm.write(f"[{seq_name}]")
        _run_sequence(seq_dir, predictor, conf_thresh=args.conf, overwrite=args.overwrite)

    logger.info("Done.")


if __name__ == "__main__":
    main()

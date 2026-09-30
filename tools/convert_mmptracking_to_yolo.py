"""Convert MMPTracking original labels to YOLO detection format.

Input layout (relative to --root):
    <split>/labels/<session>/<scene>/rgb_XXXXX_C.json   {"<id>": [l, t, r, b], ...}
    <split>/images/<session>/<scene>/rgb_XXXXX_C.jpg    (optional, see below)

Output layout (relative to --out), mirroring the input tree:
    <split>/labels/<session>/<scene>/rgb_XXXXX_C.txt    "0 cx cy w h" normalized to [0, 1]
    <split>/images/<session>/<scene>/rgb_XXXXX_C.jpg    symlink, only if --link-images
    data.yaml

Images are NOT required: when an image is missing, --img-size (default 640x360) is used for
normalization. When it exists, its real size is used. Ultralytics finds a label by replacing
"/images/" with "/labels/" in the image path, so the .txt tree can be dropped next to the
original images/ tree on another machine.

Boxes are clipped to the image; boxes smaller than --min-size px after clipping are dropped.

Usage:
    python tools/convert_mmptracking_to_yolo.py --split train
    python tools/convert_mmptracking_to_yolo.py --split validation
"""

import argparse
import glob
import json
import os
import os.path as osp
from collections import Counter

from PIL import Image

DEFAULT_ROOT = osp.join(osp.dirname(osp.abspath(__file__)), "..", "datasets", "MMPTracking")


def to_yolo(box, w, h, min_size):
    l, t, r, b = box
    l, r = max(0.0, min(l, w)), max(0.0, min(r, w))
    t, b = max(0.0, min(t, h)), max(0.0, min(b, h))
    bw, bh = r - l, b - t
    if bw < min_size or bh < min_size:
        return None
    return (l + bw / 2) / w, (t + bh / 2) / h, bw / w, bh / h


def convert_split(root, split, out, img_size, min_size, link_images):
    label_root = osp.join(root, split, "labels")
    image_root = osp.join(root, split, "images")
    json_paths = sorted(glob.glob(osp.join(label_root, "**", "*.json"), recursive=True))
    if not json_paths:
        raise SystemExit(f"No .json labels found under {label_root}")

    stats = Counter()
    per_scene = Counter()
    for json_path in json_paths:
        rel = osp.relpath(json_path, label_root)  # <session>/<scene>/rgb_XXXXX_C.json
        img_path = osp.join(image_root, rel[:-5] + ".jpg")
        if osp.exists(img_path):
            with Image.open(img_path) as im:
                w, h = im.size
            stats["size_from_image"] += 1
        else:
            w, h = img_size
            stats["size_from_arg"] += 1

        with open(json_path) as f:
            label = json.load(f)
        lines = []
        for box in label.values():
            stats["boxes"] += 1
            l, t, r, b = box
            if l < 0 or t < 0 or r > w or b > h:
                stats["clipped"] += 1
            yolo = to_yolo(box, w, h, min_size)
            if yolo is None:
                stats["dropped"] += 1
                continue
            lines.append("0 " + " ".join(f"{v:.6f}" for v in yolo))

        dst_txt = osp.join(out, split, "labels", rel[:-5] + ".txt")
        os.makedirs(osp.dirname(dst_txt), exist_ok=True)
        with open(dst_txt, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        stats["labels"] += 1
        per_scene[osp.dirname(rel)] += 1

        if link_images and osp.exists(img_path):
            dst_img = osp.join(out, split, "images", rel[:-5] + ".jpg")
            os.makedirs(osp.dirname(dst_img), exist_ok=True)
            if osp.lexists(dst_img):
                os.remove(dst_img)
            os.symlink(osp.abspath(img_path), dst_img)
            stats["linked_images"] += 1
    return stats, per_scene


def write_data_yaml(out):
    splits = {"train": "train", "val": "validation"}
    with open(osp.join(out, "data.yaml"), "w") as f:
        f.write(f"path: {out}\n")
        for key, split in splits.items():
            f.write(f"{key}: {split}/images\n")
        f.write("\nnames:\n  0: person\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=DEFAULT_ROOT, help="MMPTracking dataset root")
    p.add_argument("--split", required=True, help="split folder name, e.g. train / validation")
    p.add_argument("--out", default=None, help="output root (default: <root>/yolo)")
    p.add_argument("--img-size", type=int, nargs=2, default=(640, 360), metavar=("W", "H"),
                   help="image size used when the image file is not available")
    p.add_argument("--min-size", type=float, default=2.0, help="drop boxes narrower/shorter than this (px)")
    p.add_argument("--link-images", action="store_true", help="symlink existing images into <out>")
    args = p.parse_args()

    out = osp.abspath(args.out or osp.join(args.root, "yolo"))
    stats, per_scene = convert_split(args.root, args.split, out, args.img_size, args.min_size, args.link_images)
    for scene, n in sorted(per_scene.items()):
        print(f"  {scene}: {n} labels")
    print(f"{args.split}: labels={stats['labels']} boxes={stats['boxes']} clipped={stats['clipped']} "
          f"dropped={stats['dropped']} size_from_image={stats['size_from_image']} "
          f"size_from_arg={stats['size_from_arg']} linked_images={stats['linked_images']}")
    write_data_yaml(out)
    print(f"Wrote {osp.join(out, 'data.yaml')}")


if __name__ == "__main__":
    main()

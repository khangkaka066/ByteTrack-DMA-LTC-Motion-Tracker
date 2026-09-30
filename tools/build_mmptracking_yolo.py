"""Download MMPTracking from the Hugging Face Hub and build a flat YOLO dataset.

Hub layout (repo hungnguyen190204/MMPTracking, prefix MMPTracking/):
    <split>/images/<session>/<scene>.zip   -> <scene>/rgb_XXXXX_C.jpg
    <split>/labels/<session>/<scene>.zip   -> <scene>/rgb_XXXXX_C.json  {"<id>": [l, t, r, b]}
    split: train (sessions 63am, 64am), validation (session 64pm)

Output (--out):
    images/{train,val}/<session>_<scene>_rgb_XXXXX_C.jpg
    labels/{train,val}/<session>_<scene>_rgb_XXXXX_C.txt   "0 cx cy w h" normalized
    data.yaml

Scenes are processed one at a time: the two zips are downloaded to --tmp, frames are read
straight from the zips (no extraction), then the zips are deleted. Finished scenes are
recorded in <out>/.done so the script can be re-run to resume.

With --upload-repo, nothing is kept locally: each scene is written into one shard zip
(images/<split>/..., labels/<split>/...), uploaded to <upload-repo>/<upload-prefix>/<split>/
<session>_<scene>.zip, then deleted. Unzipping all shards into one folder gives the dataset.

The full dataset is ~103 GB (train) + ~51 GB (val) of JPEGs. Use --stride to keep every
N-th frame (consecutive frames are near-duplicates) and --cams to keep some cameras.

Usage:
    python tools/build_mmptracking_yolo.py --dry-run                 # list scenes and sizes
    python tools/build_mmptracking_yolo.py --stride 5
    python tools/build_mmptracking_yolo.py --splits validation --scenes cafe_shop_0 lobby_0
    HF_TOKEN=hf_xxx python tools/build_mmptracking_yolo.py --stride 5 \
        --upload-repo <user>/<repo>
"""

import argparse
import io
import os
import os.path as osp
import re
import shutil
import zipfile
import json

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

from convert_mmptracking_to_yolo import to_yolo

REPO_ID = "hungnguyen190204/MMPTracking"
PREFIX = "MMPTracking"
SPLIT_OUT = {"train": "train", "validation": "val"}
NAME_RE = re.compile(r"rgb_(\d+)_(\d+)\.(jpg|json)$")
DEFAULT_OUT = osp.join(osp.dirname(osp.abspath(__file__)), "..", "datasets", "MMPTracking", "yolo")


def list_scenes(api, splits, scenes):
    """Return [(split, session, scene, image_zip_size)] available on the hub."""
    items = []
    for split in splits:
        for entry in api.list_repo_tree(REPO_ID, f"{PREFIX}/{split}/images", repo_type="dataset",
                                        recursive=True):
            if not entry.path.endswith(".zip"):
                continue
            session, zip_name = entry.path.split("/")[-2:]
            scene = zip_name[:-4]
            if scenes and scene not in scenes:
                continue
            items.append((split, session, scene, entry.size))
    return sorted(items)


def download(split, kind, session, scene, tmp):
    path = hf_hub_download(REPO_ID, f"{PREFIX}/{split}/{kind}/{session}/{scene}.zip",
                           repo_type="dataset", local_dir=tmp)
    return path


def folder_writer(out):
    def write(rel, data):
        with open(osp.join(out, rel), "wb") as f:
            f.write(data)
    return write


def process_scene(split, session, scene, write, tmp, args):
    out_split = SPLIT_OUT[split]
    label_zip = download(split, "labels", session, scene, tmp)
    image_zip = download(split, "images", session, scene, tmp)

    stats = {"images": 0, "boxes": 0, "clipped": 0, "dropped": 0, "no_label": 0}
    with zipfile.ZipFile(label_zip) as lz, zipfile.ZipFile(image_zip) as iz:
        labels = {osp.basename(n)[:-5]: n for n in lz.namelist() if n.endswith(".json")}
        for member in sorted(iz.namelist()):
            m = NAME_RE.search(member)
            if not m or m.group(3) != "jpg":
                continue
            frame, cam = int(m.group(1)), int(m.group(2))
            if frame % args.stride or (args.cams and cam not in args.cams):
                continue
            data = iz.read(member)
            w, h = Image.open(io.BytesIO(data)).size

            stem = osp.basename(member)[:-4]
            lines = []
            if stem in labels:
                for l, t, r, b in json.loads(lz.read(labels[stem])).values():
                    stats["boxes"] += 1
                    if l < 0 or t < 0 or r > w or b > h:
                        stats["clipped"] += 1
                    yolo = to_yolo((l, t, r, b), w, h, args.min_size)
                    if yolo is None:
                        stats["dropped"] += 1
                        continue
                    lines.append("0 " + " ".join(f"{v:.6f}" for v in yolo))
            else:
                stats["no_label"] += 1

            name = f"{session}_{scene}_{stem}"
            write(f"images/{out_split}/{name}.jpg", data)
            write(f"labels/{out_split}/{name}.txt", ("\n".join(lines) + ("\n" if lines else "")).encode())
            stats["images"] += 1

    if not args.keep_zips:
        os.remove(label_zip)
        os.remove(image_zip)
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--tmp", default=None, help="download dir for zips (default: <out>/_zips)")
    p.add_argument("--splits", nargs="+", default=["train", "validation"], choices=list(SPLIT_OUT))
    p.add_argument("--scenes", nargs="+", default=None, help="e.g. cafe_shop_0 lobby_2 (default: all)")
    p.add_argument("--stride", type=int, default=1, help="keep frames whose index %% stride == 0")
    p.add_argument("--cams", type=int, nargs="+", default=None, help="keep only these camera ids")
    p.add_argument("--min-size", type=float, default=2.0, help="drop boxes narrower/shorter than this (px)")
    p.add_argument("--keep-zips", action="store_true")
    p.add_argument("--upload-repo", default=None,
                   help="HF dataset repo to upload per-scene shard zips to (needs HF_TOKEN with write access)")
    p.add_argument("--upload-prefix", default="yolo", help="folder inside --upload-repo")
    p.add_argument("--private", action="store_true", help="create --upload-repo as private if it does not exist")
    p.add_argument("--dry-run", action="store_true", help="only list scenes and estimated sizes")
    args = p.parse_args()

    out = osp.abspath(args.out)
    tmp = osp.abspath(args.tmp or osp.join(out, "_zips"))
    items = list_scenes(HfApi(), args.splits, set(args.scenes or []))
    keep = 1 / args.stride * (len(args.cams) / 4 if args.cams else 1)
    total = sum(s for *_, s in items)
    for split, session, scene, size in items:
        print(f"  {split:<10} {session} {scene:<18} {size / 1e9:6.2f} GB")
    print(f"{len(items)} scenes, image zips {total / 1e9:.1f} GB, "
          f"estimated output ~{total * keep / 1e9:.1f} GB (stride={args.stride}, cams={args.cams or 'all'})")
    if args.dry_run:
        return

    api = HfApi()
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(osp.join(out, sub), exist_ok=True)
    os.makedirs(tmp, exist_ok=True)
    data_yaml = osp.join(out, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"path: {'.' if args.upload_repo else out}\ntrain: images/train\nval: images/val\n\n"
                "names:\n  0: person\n")
    if args.upload_repo:
        api.create_repo(args.upload_repo, repo_type="dataset", private=args.private, exist_ok=True)
        api.upload_file(path_or_fileobj=data_yaml, path_in_repo=f"{args.upload_prefix}/data.yaml",
                        repo_id=args.upload_repo, repo_type="dataset", commit_message="Add data.yaml")

    done_path = osp.join(out, ".done")
    done = set(open(done_path).read().split()) if osp.exists(done_path) else set()
    for i, (split, session, scene, size) in enumerate(items, 1):
        key = f"{split}/{session}/{scene}"
        if key in done:
            print(f"[{i}/{len(items)}] {key}: already done, skip")
            continue
        need = size * (1 + 2 * keep if args.upload_repo else 1 + keep) * 1.1
        free = shutil.disk_usage(out).free
        if free < need:
            raise SystemExit(f"Not enough disk for {key}: need ~{need / 1e9:.1f} GB, free {free / 1e9:.1f} GB")
        print(f"[{i}/{len(items)}] {key} ({size / 1e9:.2f} GB) ...", flush=True)
        if args.upload_repo:
            shard_name = f"{session}_{scene}.zip"
            shard = osp.join(tmp, shard_name)
            with zipfile.ZipFile(shard, "w", zipfile.ZIP_STORED) as zf:
                s = process_scene(split, session, scene, zf.writestr, tmp, args)
            print(f"    uploading {shard_name} ({osp.getsize(shard) / 1e9:.2f} GB) ...", flush=True)
            api.upload_file(path_or_fileobj=shard,
                            path_in_repo=f"{args.upload_prefix}/{SPLIT_OUT[split]}/{shard_name}",
                            repo_id=args.upload_repo, repo_type="dataset",
                            commit_message=f"Add YOLO shard {SPLIT_OUT[split]}/{shard_name}")
            os.remove(shard)
        else:
            s = process_scene(split, session, scene, folder_writer(out), tmp, args)
        print(f"    images={s['images']} boxes={s['boxes']} clipped={s['clipped']} "
              f"dropped={s['dropped']} no_label={s['no_label']}", flush=True)
        with open(done_path, "a") as f:
            f.write(key + "\n")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Done. {osp.join(out, 'data.yaml')}")


if __name__ == "__main__":
    main()

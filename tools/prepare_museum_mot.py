"""Convert MuseumVisitors into the MOT/COCO layout used by tools/track_dma.py.

Each camera becomes one sequence. Only frames from the held-out splits of
MuseumVisitors_YOLO (val/test by default, which are contiguous in time and
never used for training) are exported, so detectors are benchmarked on unseen
frames.

Output layout (data_dir = datasets/museum_mot):
    annotations/<split_name>.json        COCO json with video_id/frame_id/track_id
    <split_name>/cam{c}/img1/XXXXXX.jpg  symlinks to the original frames
    <split_name>/cam{c}/gt/gt.txt        MOT gt: frame,id,x,y,w,h,1,1,1
    <split_name>/cam{c}/seqinfo.ini

Usage:
    python tools/prepare_museum_mot.py                     # val+test -> "test"
    python tools/prepare_museum_mot.py --splits test
"""
import argparse
import collections
import configparser
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_parser():
    p = argparse.ArgumentParser("Prepare MuseumVisitors for MOT evaluation")
    p.add_argument("--src", default=os.path.join(ROOT, "datasets", "MuseumVisitors"))
    p.add_argument("--yolo", default=os.path.join(ROOT, "datasets", "MuseumVisitors_YOLO"))
    p.add_argument("--out", default=os.path.join(ROOT, "datasets", "museum_mot"))
    p.add_argument("--splits", default="val,test",
                   help="comma separated MuseumVisitors_YOLO splits whose frames are exported")
    p.add_argument("--name", default="test", help="output split / json name")
    p.add_argument("--fps", type=int, default=30)
    return p


def split_frame_ranges(yolo_dir, splits):
    """{cam: (first_frame, last_frame)} covering the requested YOLO splits."""
    frames = collections.defaultdict(set)
    for split in splits:
        for f in os.listdir(os.path.join(yolo_dir, "images", split)):
            m = re.match(r"cam(\d+)_.*_(\d+)\.jpg$", f)
            if m:
                frames[int(m[1])].add(int(m[2]))
    ranges = {}
    for cam, fs in frames.items():
        lo, hi = min(fs), max(fs)
        missing = (hi - lo + 1) - len(fs)
        if missing:
            print(f"[warn] cam{cam}: {missing} frame(s) missing in {lo}..{hi}")
        ranges[cam] = (lo, hi)
    return ranges


def main():
    args = make_parser().parse_args()
    ranges = split_frame_ranges(args.yolo, args.splits.split(","))

    # annotations.dat: id,cam,frame,x,y,w,h (full),x,y,w,h (visible),gaze..,path,opera,group
    # The path column says folder "4/" for camera 3 while frames live in frames/3,
    # so index by the camera column and resolve images from frames/<cam>.
    anns = collections.defaultdict(list)       # (cam, frame) -> [(tid, x, y, w, h)]
    paths = {}                                 # (cam, frame) -> relative image path
    with open(os.path.join(args.src, "annotations.dat")) as fh:
        for line in fh:
            r = line.strip().split(",")
            if len(r) < 14:
                continue
            cam = int(r[1])
            path = "/".join([str(cam)] + r[13].split("/")[1:])
            frame = int(r[2])
            anns[(cam, frame)].append((int(r[0]), *map(float, r[3:7])))
            paths[(cam, frame)] = path

    frames_root = os.path.join(args.src, "frames")
    on_disk = {}
    for cam in ranges:
        for dirpath, _, files in os.walk(os.path.join(frames_root, str(cam))):
            for f in files:
                m = re.search(r"_(\d+)\.jpg$", f)
                if m:
                    on_disk[(cam, int(m[1]))] = os.path.join(dirpath, f)

    coco = {"images": [], "annotations": [], "videos": [],
            "categories": [{"id": 1, "name": "pedestrian"}]}
    img_id = ann_id = 0
    for vid, cam in enumerate(sorted(ranges), start=1):
        lo, hi = ranges[cam]
        seq = f"cam{cam}"
        seq_dir = os.path.join(args.out, args.name, seq)
        os.makedirs(os.path.join(seq_dir, "img1"), exist_ok=True)
        os.makedirs(os.path.join(seq_dir, "gt"), exist_ok=True)
        coco["videos"].append({"id": vid, "file_name": seq})

        gt_lines = []
        n_frames = 0
        width = height = None
        for orig in range(lo, hi + 1):
            src = on_disk.get((cam, orig))
            if src is None and (cam, orig) in paths:
                src = os.path.join(frames_root, paths[(cam, orig)])
            if src is None or not os.path.isfile(src):
                print(f"[warn] {seq}: frame {orig} not found on disk, skipped")
                continue
            if width is None:
                import cv2
                height, width = cv2.imread(src).shape[:2]
            n_frames += 1
            frame_id = n_frames
            dst = os.path.join(seq_dir, "img1", f"{frame_id:06d}.jpg")
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)

            img_id += 1
            coco["images"].append({
                "id": img_id, "file_name": f"{seq}/img1/{frame_id:06d}.jpg",
                "frame_id": frame_id, "video_id": vid, "orig_frame": orig,
                "prev_image_id": img_id - 1 if frame_id > 1 else -1,
                "next_image_id": img_id + 1 if orig < hi else -1,
                "height": height, "width": width,
            })
            seen = set()
            for tid, x, y, w, h in anns.get((cam, orig), []):
                if tid in seen or w <= 0 or h <= 0:
                    continue
                seen.add(tid)
                ann_id += 1
                coco["annotations"].append({
                    "id": ann_id, "category_id": 1, "image_id": img_id,
                    "track_id": tid, "bbox": [x, y, w, h], "conf": 1.0,
                    "iscrowd": 0, "area": w * h,
                })
                gt_lines.append(f"{frame_id},{tid},{x:.1f},{y:.1f},{w:.1f},{h:.1f},1,1,1\n")

        coco["images"][-1]["next_image_id"] = -1
        with open(os.path.join(seq_dir, "gt", "gt.txt"), "w") as fh:
            fh.writelines(gt_lines)
        ini = configparser.ConfigParser()
        ini.optionxform = str
        ini["Sequence"] = {
            "name": seq, "imDir": "img1", "frameRate": str(args.fps),
            "seqLength": str(n_frames), "imWidth": str(width),
            "imHeight": str(height), "imExt": ".jpg",
        }
        with open(os.path.join(seq_dir, "seqinfo.ini"), "w") as fh:
            ini.write(fh, space_around_delimiters=False)
        print(f"{seq}: frames {lo}..{hi} -> {n_frames} images, {len(gt_lines)} boxes")

    os.makedirs(os.path.join(args.out, "annotations"), exist_ok=True)
    out_json = os.path.join(args.out, "annotations", f"{args.name}.json")
    with open(out_json, "w") as fh:
        json.dump(coco, fh)
    print(f"wrote {out_json}: {len(coco['images'])} images, {len(coco['annotations'])} boxes")


if __name__ == "__main__":
    main()

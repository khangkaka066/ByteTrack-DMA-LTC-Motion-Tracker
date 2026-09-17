import argparse
import os
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm


def make_parser():
    parser = argparse.ArgumentParser("SportMOT ReID patch generator")
    parser.add_argument("--data-path", default="datasets/sportmot", help="SportMOT dataset root")
    parser.add_argument("--save-path", default="datasets", help="directory to save ReID dataset")
    parser.add_argument("--dataset-name", default="SportMOT-ReID", help="output dataset folder name")
    parser.add_argument("--train-split", default="train", help="SportMOT split used for bounding_box_train")
    parser.add_argument("--val-split", default="val", help="SportMOT split used for query and bounding_box_test")
    parser.add_argument("--min-visibility", type=float, default=0.0, help="minimum MOT visibility ratio")
    parser.add_argument("--category-id", type=int, default=1, help="MOT category id to keep")
    parser.add_argument("--no-active-filter", action="store_true", help="do not require active/confidence column == 1")
    parser.add_argument("--query-per-id", type=int, default=1, help="number of validation crops per id saved to query")
    parser.add_argument("--ext", default=".jpg", choices=[".jpg", ".png", ".bmp"], help="output patch extension")
    parser.add_argument("--max-frames-per-seq", type=int, default=0, help="debug limit; 0 means process all frames")
    return parser


def read_mot_gt(gt_path, active_filter=True, category_id=1, min_visibility=0.0):
    rows = []
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = [float(x.strip()) for x in line.split(",")]
            if len(values) < 6:
                continue
            while len(values) < 9:
                values.append(1.0)
            frame_id, track_id, x, y, w, h, active, category, visibility = values[:9]
            if active_filter and int(active) != 1:
                continue
            if category_id is not None and int(category) != category_id:
                continue
            if visibility < min_visibility:
                continue
            rows.append((int(frame_id), int(track_id), x, y, x + w, y + h))
    return rows


def list_sequences(split_dir):
    if not os.path.isdir(split_dir):
        return []
    seqs = []
    for seq in sorted(os.listdir(split_dir)):
        seq_dir = os.path.join(split_dir, seq)
        if os.path.isdir(os.path.join(seq_dir, "img1")) and os.path.isfile(os.path.join(seq_dir, "gt", "gt.txt")):
            seqs.append(seq)
    return seqs


def clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width, int(round(x2))))
    y2 = max(0, min(height, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def collect_sequence_ids(split_dir, seqs, active_filter, category_id, min_visibility):
    seq_offsets = {}
    id_offset = 0
    for seq in seqs:
        gt_path = os.path.join(split_dir, seq, "gt", "gt.txt")
        rows = read_mot_gt(gt_path, active_filter, category_id, min_visibility)
        ids = [track_id for _, track_id, *_ in rows]
        seq_offsets[seq] = id_offset
        if ids:
            id_offset += max(ids) + 1
    return seq_offsets


def save_patch(img, box, save_dir, global_id, cam_id, frame_id, seq, ext):
    height, width = img.shape[:2]
    clamped = clamp_box(box, width, height)
    if clamped is None:
        return False
    x1, y1, x2, y2 = clamped
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return False

    # Market1501 parser reads pid and camid from "<pid>_c<camid>...".
    filename = "{:07d}_c{:04d}_f{:07d}_{}.{}".format(
        global_id,
        cam_id,
        frame_id,
        seq,
        ext.lstrip("."),
    )
    return cv2.imwrite(os.path.join(save_dir, filename), patch)


def process_split(split_dir, seqs, save_dir, query_dir, seq_offsets, args, make_query=False):
    saved = 0
    skipped = 0
    query_counts = defaultdict(int)

    for cam_id, seq in enumerate(tqdm(seqs, desc=os.path.basename(split_dir)), start=1):
        seq_dir = os.path.join(split_dir, seq)
        img_dir = os.path.join(seq_dir, "img1")
        gt_path = os.path.join(seq_dir, "gt", "gt.txt")
        rows = read_mot_gt(
            gt_path,
            active_filter=not args.no_active_filter,
            category_id=args.category_id,
            min_visibility=args.min_visibility,
        )
        by_frame = defaultdict(list)
        for frame_id, track_id, x1, y1, x2, y2 in rows:
            by_frame[frame_id].append((track_id, x1, y1, x2, y2))

        for frame_id in sorted(by_frame):
            if args.max_frames_per_seq > 0 and frame_id > args.max_frames_per_seq:
                continue
            img_path = os.path.join(img_dir, "{:06d}.jpg".format(frame_id))
            img = cv2.imread(img_path)
            if img is None:
                skipped += len(by_frame[frame_id])
                continue

            for track_id, x1, y1, x2, y2 in by_frame[frame_id]:
                global_id = seq_offsets[seq] + track_id + 1
                target_dir = save_dir
                if make_query and query_counts[global_id] < args.query_per_id:
                    target_dir = query_dir
                    query_counts[global_id] += 1
                if save_patch(img, (x1, y1, x2, y2), target_dir, global_id, cam_id, frame_id, seq, args.ext):
                    saved += 1
                else:
                    skipped += 1

    return saved, skipped, len(query_counts)


def main(args):
    output_root = os.path.join(args.save_path, args.dataset_name)
    train_dir = os.path.join(output_root, "bounding_box_train")
    gallery_dir = os.path.join(output_root, "bounding_box_test")
    query_dir = os.path.join(output_root, "query")
    for path in (train_dir, gallery_dir, query_dir):
        os.makedirs(path, exist_ok=True)

    train_split_dir = os.path.join(args.data_path, args.train_split)
    val_split_dir = os.path.join(args.data_path, args.val_split)
    train_seqs = list_sequences(train_split_dir)
    val_seqs = list_sequences(val_split_dir)

    train_offsets = collect_sequence_ids(
        train_split_dir,
        train_seqs,
        active_filter=not args.no_active_filter,
        category_id=args.category_id,
        min_visibility=args.min_visibility,
    )
    train_id_count = 0
    if train_offsets:
        last_seq = train_seqs[-1]
        last_rows = read_mot_gt(
            os.path.join(train_split_dir, last_seq, "gt", "gt.txt"),
            active_filter=not args.no_active_filter,
            category_id=args.category_id,
            min_visibility=args.min_visibility,
        )
        train_id_count = train_offsets[last_seq] + (max([r[1] for r in last_rows]) + 1 if last_rows else 0)

    val_offsets = collect_sequence_ids(
        val_split_dir,
        val_seqs,
        active_filter=not args.no_active_filter,
        category_id=args.category_id,
        min_visibility=args.min_visibility,
    )
    val_offsets = {seq: offset + train_id_count for seq, offset in val_offsets.items()}

    train_saved, train_skipped, _ = process_split(
        train_split_dir, train_seqs, train_dir, query_dir, train_offsets, args, make_query=False
    )
    val_saved, val_skipped, query_ids = process_split(
        val_split_dir, val_seqs, gallery_dir, query_dir, val_offsets, args, make_query=True
    )

    print("Saved train patches: {}".format(train_saved))
    print("Saved gallery patches: {}".format(val_saved))
    print("Saved query identities: {}".format(query_ids))
    print("Skipped boxes/images: {}".format(train_skipped + val_skipped))
    print("Output: {}".format(output_root))


if __name__ == "__main__":
    main(make_parser().parse_args())

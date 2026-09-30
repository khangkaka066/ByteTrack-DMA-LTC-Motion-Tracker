"""Visualize & compare MMPTracking original labels vs retrieval labels.

Label format (both): {"<id>": [l, t, r, b], ...}

Layout expected (relative to --root):
    <scene>/rgb_XXXXX_C.json                          original labels
    <scene>_image/<scene>/rgb_XXXXX_C.jpg             images
    <scene>_retrieval_label/<scene>/rgb_XXXXX_C.json  retrieval labels

Usage:
    # Statistics: how often do the two label sets differ?
    python tools/visualize_mmptracking.py compare

    # Interactive viewer: original | retrieval_label side by side
    #   d / space: next, a: prev, n: next frame that differs, q / esc: quit
    python tools/visualize_mmptracking.py view --cam 1 --start 100

    # Dump side-by-side images to disk instead of showing a window
    python tools/visualize_mmptracking.py view --cam 1 --save-dir /tmp/vis --only-diff
"""

import argparse
import json
import os
import os.path as osp
import re
from collections import Counter, defaultdict

import cv2
import numpy as np

DEFAULT_ROOT = osp.join(osp.dirname(osp.abspath(__file__)), "..", "datasets", "MMPTracking")
NAME_RE = re.compile(r"rgb_(\d+)_(\d+)\.json$")


def get_paths(root, scene):
    return (
        osp.join(root, scene),
        osp.join(root, f"{scene}_retrieval_label", scene),
        osp.join(root, f"{scene}_image", scene),
    )


def load_label(path):
    if not osp.exists(path):
        return None
    with open(path) as f:
        return {str(k): tuple(int(round(x)) for x in v) for k, v in json.load(f).items()}


def list_frames(orig_dir, cam=None):
    frames = []
    for name in os.listdir(orig_dir):
        m = NAME_RE.match(name)
        if m and (cam is None or int(m.group(2)) == cam):
            frames.append((int(m.group(2)), int(m.group(1)), name))
    return [name for _, _, name in sorted(frames)]


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def diff_frame(orig, retr):
    """Return a dict describing how retr differs from orig for one frame."""
    orig_boxes = Counter(orig.values())
    retr_boxes = Counter(retr.values())
    # id -> box matches under the same id
    same_id_same_box = [i for i in orig if i in retr and orig[i] == retr[i]]
    # orig id -> retr ids that carry the exact same box
    box_to_retr_ids = defaultdict(list)
    for i, b in retr.items():
        box_to_retr_ids[b].append(i)
    id_map = {i: box_to_retr_ids.get(b, []) for i, b in orig.items()}
    return {
        "identical": orig == retr,
        "same_id_set": set(orig) == set(retr),
        "same_box_multiset": orig_boxes == retr_boxes,
        "same_box_set": set(orig_boxes) == set(retr_boxes),
        "n_orig": len(orig),
        "n_retr": len(retr),
        "n_same_id_same_box": len(same_id_same_box),
        "boxes_only_orig": list((orig_boxes - retr_boxes).elements()),
        "boxes_only_retr": list((retr_boxes - orig_boxes).elements()),
        "retr_dup_boxes": {b: ids for b, ids in box_to_retr_ids.items() if len(ids) > 1},
        "ids_only_orig": sorted(set(orig) - set(retr), key=int),
        "ids_only_retr": sorted(set(retr) - set(orig), key=int),
        "id_map": id_map,
    }


# --------------------------------------------------------------------------- compare
def cmd_compare(args):
    orig_dir, retr_dir, _ = get_paths(args.root, args.scene)
    names = list_frames(orig_dir, args.cam)
    stats = Counter()
    examples = defaultdict(list)
    per_cam = defaultdict(Counter)
    id_pairs = Counter()  # (orig_id, retr_id) co-occurrence on exact same box

    for name in names:
        cam = int(NAME_RE.match(name).group(2))
        orig = load_label(osp.join(orig_dir, name))
        retr = load_label(osp.join(retr_dir, name))
        stats["frames"] += 1
        per_cam[cam]["frames"] += 1
        if retr is None:
            stats["missing_retrieval_file"] += 1
            continue
        d = diff_frame(orig, retr)
        stats["boxes_orig"] += d["n_orig"]
        stats["boxes_retr"] += d["n_retr"]
        stats["boxes_same_id_same_box"] += d["n_same_id_same_box"]
        for oid, rids in d["id_map"].items():
            for rid in rids:
                id_pairs[(oid, rid)] += 1
        checks = {
            "identical": d["identical"],
            "same_id_set": d["same_id_set"],
            "same_box_set": d["same_box_set"],
            "same_box_multiset": d["same_box_multiset"],
            "retr_has_duplicate_boxes": bool(d["retr_dup_boxes"]),
            "retr_has_new_boxes": bool(d["boxes_only_retr"]),
            "retr_missing_boxes": bool(d["boxes_only_orig"]),
        }
        for k, v in checks.items():
            if v:
                stats[k] += 1
                per_cam[cam][k] += 1
                if len(examples[k]) < 3 and k not in ("identical", "same_box_set"):
                    examples[k].append(name)
        if not d["identical"] and len(examples["not_identical"]) < 3:
            examples["not_identical"].append(name)

    n = max(stats["frames"], 1)
    print(f"Scene: {args.scene}  cam: {args.cam or 'all'}  frames: {stats['frames']}")
    if stats["missing_retrieval_file"]:
        print(f"  missing retrieval files: {stats['missing_retrieval_file']}")
    for k in ("identical", "same_id_set", "same_box_set", "same_box_multiset",
              "retr_has_duplicate_boxes", "retr_has_new_boxes", "retr_missing_boxes"):
        print(f"  {k:<26} {stats[k]:>7} / {n}  ({100 * stats[k] / n:5.1f}%)")
    bo = max(stats["boxes_orig"], 1)
    print(f"  boxes: orig={stats['boxes_orig']}  retr={stats['boxes_retr']}  "
          f"same id & same box={stats['boxes_same_id_same_box']} "
          f"({100 * stats['boxes_same_id_same_box'] / bo:.1f}% of orig)")

    print("\nPer camera (identical / same_box_set / dup_boxes):")
    for cam in sorted(per_cam):
        c = per_cam[cam]
        print(f"  cam {cam}: {c['identical']}/{c['frames']}  "
              f"{c['same_box_set']}/{c['frames']}  {c['retr_has_duplicate_boxes']}/{c['frames']}")

    print("\nMost frequent orig_id -> retr_id pairs (same exact box):")
    by_orig = defaultdict(list)
    for (o, r), c in id_pairs.items():
        by_orig[o].append((c, r))
    for o in sorted(by_orig, key=int):
        top = sorted(by_orig[o], reverse=True)[:4]
        print(f"  orig {o:>3} -> " + ", ".join(f"{r}({c})" for c, r in top))

    print("\nExamples:")
    for k, v in examples.items():
        print(f"  {k}: {', '.join(v)}")


# --------------------------------------------------------------------------- view
def color_for(tid):
    rng = np.random.RandomState(int(tid) * 7 + 3)
    return tuple(int(c) for c in rng.randint(40, 255, 3))


def draw(img, label, title, highlight=None):
    """highlight: set of boxes to draw thick in red (differences)."""
    img = img.copy()
    h, w = img.shape[:2]
    highlight = highlight or set()
    for tid, (l, t, r, b) in sorted(label.items(), key=lambda kv: int(kv[0])):
        c = color_for(tid)
        diff = (l, t, r, b) in highlight
        cv2.rectangle(img, (l, t), (r, b), (0, 0, 255) if diff else c, 3 if diff else 2)
        txt = f"{tid}"
        ty = max(t, 0) + 16 if t < 16 else t - 4
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tx = min(max(l, 0), w - tw - 2)
        cv2.rectangle(img, (tx, ty - th - 3), (tx + tw + 2, ty + 3), c, -1)
        cv2.putText(img, txt, (tx + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    bar = np.zeros((26, w, 3), np.uint8)
    cv2.putText(bar, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return np.vstack([bar, img])


def render(name, orig, retr, img):
    d = diff_frame(orig, retr) if retr is not None else None
    retr = retr or {}
    # A box is "different" in a panel if it's not present in the other panel under the same id.
    diff_orig = {b for i, b in orig.items() if retr.get(i) != b}
    diff_retr = {b for i, b in retr.items() if orig.get(i) != b}
    left = draw(img, orig, f"original  {name}  n={len(orig)}", diff_orig)
    right = draw(img, retr, f"retrieval_label  n={len(retr)}", diff_retr)
    sep = np.full((left.shape[0], 4, 3), 255, np.uint8)
    canvas = np.hstack([left, sep, right])

    # Info panel below the images
    lines = []
    if d is None:
        lines.append("retrieval label file MISSING")
    else:
        lines.append(f"identical={d['identical']}  same_id_set={d['same_id_set']}  "
                     f"same_box_set={d['same_box_set']}  same_id&box={d['n_same_id_same_box']}/{d['n_orig']}")
        mapping = "  ".join(f"{o}->{','.join(r) or '?'}" for o, r in
                            sorted(d["id_map"].items(), key=lambda kv: int(kv[0])))
        lines.append(f"id map (orig->retr by same box): {mapping}")
        if d["retr_dup_boxes"]:
            lines.append("retr duplicate boxes: " + "  ".join(
                f"ids {','.join(ids)}" for ids in d["retr_dup_boxes"].values()))
        if d["ids_only_orig"] or d["ids_only_retr"]:
            lines.append(f"ids only in orig: {d['ids_only_orig']}   ids only in retr: {d['ids_only_retr']}")
        if d["boxes_only_orig"] or d["boxes_only_retr"]:
            lines.append(f"boxes only in orig: {len(d['boxes_only_orig'])}   "
                         f"boxes only in retr: {len(d['boxes_only_retr'])}")
    lines.append("red = box whose id/coords differ from other panel | d/space next, a prev, n next-diff, q quit")
    panel = np.zeros((22 * len(lines) + 8, canvas.shape[1], 3), np.uint8)
    for k, line in enumerate(lines):
        cv2.putText(panel, line, (6, 20 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return np.vstack([canvas, panel]), d


def cmd_view(args):
    orig_dir, retr_dir, img_dir = get_paths(args.root, args.scene)
    names = list_frames(orig_dir, args.cam)
    if args.start:
        names = [n for n in names if int(NAME_RE.match(n).group(1)) >= args.start]
    if not names:
        raise SystemExit("No frames found")

    def load(idx):
        name = names[idx]
        orig = load_label(osp.join(orig_dir, name))
        retr = load_label(osp.join(retr_dir, name))
        img = cv2.imread(osp.join(img_dir, name.replace(".json", ".jpg")))
        if img is None:
            img = np.zeros((360, 640, 3), np.uint8)
        return render(name, orig, retr, img)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        saved = 0
        for idx in range(0, len(names), args.step):
            vis, d = load(idx)
            if args.only_diff and d is not None and d["identical"]:
                continue
            cv2.imwrite(osp.join(args.save_dir, names[idx].replace(".json", ".jpg")), vis)
            saved += 1
            if args.max and saved >= args.max:
                break
        print(f"Saved {saved} images to {args.save_dir}")
        return

    idx = 0
    win = "original | retrieval_label"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        vis, _ = load(idx)
        if args.scale != 1.0:
            vis = cv2.resize(vis, None, fx=args.scale, fy=args.scale)
        cv2.imshow(win, vis)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord("d"), ord(" "), 83):
            idx = min(idx + args.step, len(names) - 1)
        elif key in (ord("a"), 81):
            idx = max(idx - args.step, 0)
        elif key == ord("n"):
            j = idx + 1
            while j < len(names):
                _, d = load(j)
                if d is None or not d["identical"]:
                    break
                j += 1
            idx = min(j, len(names) - 1)
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=DEFAULT_ROOT, help="MMPTracking dataset root")
    p.add_argument("--scene", default="cafe_shop_0")
    p.add_argument("--cam", type=int, default=None, help="camera index (1-4); default all")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("compare", help="print statistics of differences")
    v = sub.add_parser("view", help="side-by-side viewer")
    v.add_argument("--start", type=int, default=0, help="first frame number")
    v.add_argument("--step", type=int, default=1)
    v.add_argument("--scale", type=float, default=1.5, help="display scale")
    v.add_argument("--save-dir", default=None, help="write images instead of opening a window")
    v.add_argument("--only-diff", action="store_true", help="with --save-dir: skip identical frames")
    v.add_argument("--max", type=int, default=0, help="with --save-dir: max images to save (0 = all)")
    args = p.parse_args()
    {"compare": cmd_compare, "view": cmd_view}[args.cmd](args)


if __name__ == "__main__":
    main()

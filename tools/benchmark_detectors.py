"""Benchmark detectors under plain ByteTrack (detection + tracking only).

Every model is run through tools/track_dma.py on the MuseumVisitors MOT split
(tools/prepare_museum_mot.py) with ReID / DMA / ML / LTC / xLSTM / sLSTM all
disabled, the same input size, batch size 1 and the same tracker settings.
Per-model summary.json files are collected into a CSV/JSON table and plotted
with tools/plot_detector_benchmark.py.

Usage:
    python tools/prepare_museum_mot.py              # once
    python tools/benchmark_detectors.py             # all models in pretrained/
    python tools/benchmark_detectors.py --models yolov8n,yolox-nano --size 640x640
    python tools/benchmark_detectors.py --skip-run  # only re-collect + re-plot
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "exps", "example", "museum", "yolox_nano_museum.py")

# name -> (kind, weights). "yolox" uses the exp's YOLOX-nano model, "ultra" uses --yolo-weights.
MODELS = {
    "yolox-nano": ("yolox", "pretrained/yolox-nano.pth"),
    "yolov8n": ("ultra", "pretrained/yolov8n.pt"),
    "yolov10n": ("ultra", "pretrained/yolov10n.pt"),
    "yolo11n": ("ultra", "pretrained/yolo11n.pt"),
    "yolo12n": ("ultra", "pretrained/yolo12n.pt"),
    "yolo26n": ("ultra", "pretrained/yolo26n.pt"),
}


def make_parser():
    p = argparse.ArgumentParser("Detector benchmark for ByteTrack")
    p.add_argument("--models", default=",".join(MODELS), help="comma separated subset of: " + ",".join(MODELS))
    p.add_argument("--size", default="416x640", help="input HxW shared by all detectors")
    p.add_argument("--tag", default="museum", help="output folder suffix: YOLOX_outputs/detbench_<tag>/")
    p.add_argument("--no-fp16", action="store_true", help="run in FP32 instead of FP16")
    p.add_argument("--conf", type=float, default=0.01, help="detector score threshold (before tracking)")
    p.add_argument("--nms", type=float, default=0.7)
    p.add_argument("--track-thresh", type=float, default=0.6)
    p.add_argument("--track-buffer", type=int, default=30)
    p.add_argument("--match-thresh", type=float, default=0.9)
    p.add_argument("--min-box-area", type=float, default=100)
    p.add_argument("--warmup-iters", type=int, default=30)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save-videos", action="store_true")
    p.add_argument("--skip-run", action="store_true", help="only collect existing results and plot")
    p.add_argument("--force", action="store_true", help="rerun models that already have summary.json")
    return p


def run_model(name, args, h, w):
    kind, weights = MODELS[name]
    expn = f"detbench_{args.tag}/{name}"
    summary = os.path.join(ROOT, "YOLOX_outputs", expn, "summary.json")
    if os.path.isfile(summary) and not args.force:
        print(f"[skip] {name}: {summary} exists (use --force to rerun)")
        return
    cmd = [
        sys.executable, "tools/track_dma.py",
        "-f", EXP, "-expn", expn, "-b", "1", "-d", "1",
        "--conf", str(args.conf), "--nms", str(args.nms),
        "--track_thresh", str(args.track_thresh), "--track_buffer", str(args.track_buffer),
        "--match_thresh", str(args.match_thresh), "--min-box-area", str(args.min_box_area),
        "--warmup-iters", str(args.warmup_iters),
    ]
    if kind == "yolox":
        cmd += ["-c", weights, "--fuse"]
    else:
        cmd += ["--yolo-weights", weights]
    if not args.no_fp16:
        cmd.append("--fp16")
    if args.save_videos:
        cmd.append("--save-videos")
    cmd += ["test_h", str(h), "test_w", str(w), "data_num_workers", str(args.workers)]

    print(f"\n=== {name} ===\n" + " ".join(cmd))
    t0 = time.time()
    ret = subprocess.run(cmd, cwd=ROOT)
    if ret.returncode != 0 or not os.path.isfile(summary):
        print(f"[fail] {name}: exit code {ret.returncode}, no summary.json")
    else:
        print(f"[done] {name} in {time.time() - t0:.0f}s")


def collect(names, tag):
    rows = []
    for name in names:
        path = os.path.join(ROOT, "YOLOX_outputs", f"detbench_{tag}", name, "summary.json")
        if not os.path.isfile(path):
            continue
        with open(path) as fh:
            s = json.load(fh)
        hota = s.get("hota") or {}
        mot = s.get("motmetrics") or {}
        speed = s.get("speed") or {}
        det = s.get("detection") or {}
        cx = s.get("detector_complexity") or {}
        rows.append({
            "model": name,
            "HOTA": hota.get("HOTA"), "DetA": hota.get("DetA"), "AssA": hota.get("AssA"),
            "MOTA": hota.get("MOTA"), "IDF1": hota.get("IDF1"),
            "IDSW": mot.get("num_switches"),
            "AP50_95": det.get("AP50_95"), "AP50": det.get("AP50"),
            "forward_ms": speed.get("forward_ms"), "track_ms": speed.get("track_ms"),
            "total_ms": speed.get("total_ms"), "fps": speed.get("fps"),
            "params_M": cx.get("params_M"), "flops_G": cx.get("flops_G"),
            "test_size": "x".join(map(str, s.get("test_size") or [])), "fp16": s.get("fp16"),
        })
    return rows


def main():
    args = make_parser().parse_args()
    names = [n.strip() for n in args.models.split(",") if n.strip()]
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        sys.exit(f"unknown model(s): {unknown}; choose from {list(MODELS)}")
    h, w = (int(v) for v in args.size.lower().split("x"))

    if not os.path.isfile(os.path.join(ROOT, "datasets", "museum_mot", "annotations", "test.json")):
        sys.exit("datasets/museum_mot not found - run `python tools/prepare_museum_mot.py` first")

    if not args.skip_run:
        for name in names:
            run_model(name, args, h, w)

    rows = collect(names, args.tag)
    if not rows:
        sys.exit("no results collected")
    out_dir = os.path.join(ROOT, "YOLOX_outputs", f"detbench_{args.tag}")
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(rows, fh, indent=2)

    hdr = f"{'model':<12}{'HOTA':>7}{'IDF1':>7}{'MOTA':>7}{'AP':>7}{'IDSW':>7}{'fwd ms':>8}{'trk ms':>8}{'FPS':>7}{'GFLOPs':>8}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in sorted(rows, key=lambda r: -(r["HOTA"] or 0)):
        print(f"{r['model']:<12}{r['HOTA']:7.2f}{r['IDF1']:7.2f}{r['MOTA']:7.2f}{r['AP50_95']:7.2f}"
              f"{int(r['IDSW'] or 0):7d}{r['forward_ms']:8.2f}{r['track_ms']:8.2f}{r['fps']:7.1f}{r['flops_G']:8.2f}")
    print(f"\nresults: {out_dir}/results.csv")

    from plot_detector_benchmark import plot
    plot(os.path.join(out_dir, "results.json"), out_dir)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()

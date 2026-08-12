#!/usr/bin/env python3
"""Benchmark FPS / Latency / FLOPs / Params across DMA & LTC pipeline configs.

Runs tools/track_dma.py once per configuration (Baseline, +DMA, +LTC,
+LTC+DMA), reads the resulting summary.json (speed + detector complexity
saved by track_dma.py), adds LTC / GBM extra complexity on top, and prints
a markdown table:

    | Model      | FPS | Latency | FLOPs | Params |

FLOPs/Params for the GBM fusion model are not directly comparable to neural
FLOPs (tree ensemble, not matmuls) so they're reported separately as
"<detector value> (+GBM: N trees)".

Usage:
    python3 tools/benchmark_pipeline_table.py --dataset mot17
    python3 tools/benchmark_pipeline_table.py --dataset sportmot
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATASETS = {
    "mot17": {
        "exp_file": "exps/example/mot/yolox_x_mot17_half.py",
        "ckpt": "pretrained/bytetrack_x_mot17.pth.tar",
        "fast_reid_config": "fast-reid/configs/MOT17/sbs_S50.yml",
        "fast_reid_weights": "reid_weights/mot17_sbs_S50.pth",
        "gbm_weights": "dma_weights/gbm_mot17/noval/dma_gbm_tuned.gbm",
        "ltc_ckpt": None,
        "reid_weight": 0.1,
    },
    "sportmot": {
        "exp_file": "exps/example/sportmot/yolox_x_sportsmot.py",
        "ckpt": "pretrained/yolox_x_sports_mix.pth.tar",
        "fast_reid_config": "fast-reid/configs/SportMOT/sbs_S50.yml",
        "fast_reid_weights": "reid_weights/sportmot_sbs_S50.pth",
        "gbm_weights": "dma_weights/gbm_sportmot/noval/dma_gbm_tuned.gbm",
        "ltc_ckpt": "ltc_weights/ltc_motion_sportsmot.pth",
        "reid_weight": 0.856,
    },
}


def make_parser():
    parser = argparse.ArgumentParser("Pipeline module benchmark (FPS/Latency/FLOPs/Params)")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()))
    parser.add_argument("--output-root", default="YOLOX_outputs", help="track_dma.py -expn output root")
    parser.add_argument("--tag", default="bench", help="prefix for the -expn experiment name")
    parser.add_argument("--md-out", default=None, help="path to write the markdown table")
    return parser


def run_track_dma(exp_file, ckpt, exp_name, extra_args, output_root):
    cmd = [
        sys.executable, os.path.join(ROOT, "tools", "track_dma.py"),
        "-f", os.path.join(ROOT, exp_file),
        "-c", os.path.join(ROOT, ckpt),
        "-expn", exp_name,
        "-b", "1", "-d", "1", "--fuse", "--fp16",
    ] + extra_args + ["output_dir", os.path.join(ROOT, output_root)]
    print("=== Running: {} ===".format(" ".join(cmd)))
    subprocess.run(cmd, check=True, cwd=ROOT)
    summary_path = os.path.join(ROOT, output_root, exp_name, "summary.json")
    with open(summary_path) as f:
        return json.load(f)


def ltc_complexity(ckpt_path):
    import torch
    from thop import profile
    from yolox.tracker.ltc_motion import LtcMotionResidual

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    history_len = int(ckpt.get("history_len", 16))
    input_dim = int(ckpt.get("input_dim", 12))
    hidden_size = int(ckpt.get("hidden_size", 128))
    num_layers = int(ckpt.get("num_layers", 2))

    model = LtcMotionResidual(
        input_dim=input_dim, history_len=history_len,
        hidden_size=hidden_size, num_layers=num_layers,
    )
    model.load_state_dict(state_dict)
    model.eval()
    dummy = torch.randn(1, history_len, input_dim)
    flops, params = profile(model, inputs=(dummy,), verbose=False)
    return params / 1e6, flops / 1e9


def reid_complexity(config_file, weights_path):
    """Params/FLOPs of the fast-reid embedding extractor for a single (1,3,pH,pW) crop."""
    import torch
    from thop import profile
    from yolox.tracker.reid import FastReIDExtractor

    extractor = FastReIDExtractor(config_file, weights_path, device="cpu")
    model = extractor.model.model.eval().to("cpu")
    pH, pW = extractor.model.pH, extractor.model.pW
    dummy = torch.randn(1, 3, pH, pW)
    flops, params = profile(model, inputs=(dummy,), verbose=False)
    return params / 1e6, flops / 1e9


def gbm_complexity(ckpt_path):
    from yolox.DMA.model_gbm import DynamicWeightGBM

    model, _ = DynamicWeightGBM.load(ckpt_path)
    booster = model.booster
    num_trees = booster.num_trees()
    try:
        df = booster.trees_to_dataframe()
        num_leaves = int(df["split_feature"].isna().sum())
    except Exception:
        num_leaves = None
    return num_trees, num_leaves


def fmt_flops(base_g, extra_note=None):
    s = "{:.2f} G".format(base_g)
    if extra_note:
        s += " ({})".format(extra_note)
    return s


def fmt_params(base_m, extra_note=None):
    s = "{:.2f} M".format(base_m)
    if extra_note:
        s += " ({})".format(extra_note)
    return s


def main():
    args = make_parser().parse_args()
    cfg = DATASETS[args.dataset]

    common_reid_args = [
        "--with-reid", "--fast-reid",
        "--fast-reid-config", os.path.join(ROOT, cfg["fast_reid_config"]),
        "--fast-reid-weights", os.path.join(ROOT, cfg["fast_reid_weights"]),
    ]

    rows = []  # (label, summary_json, extra_flops_g, extra_params_m, note)

    def add_row(label, extra_args, extra_flops_g=0.0, extra_params_m=0.0, note=None):
        exp_name = "{}_{}_{}".format(args.tag, args.dataset, label.lower().replace("+", "").replace(" ", "_"))
        summary = run_track_dma(cfg["exp_file"], cfg["ckpt"], exp_name, extra_args, args.output_root)
        rows.append((label, summary, extra_flops_g, extra_params_m, note))

    # Baseline: pure IoU + Kalman motion, no ReID/DMA/LTC.
    add_row("Baseline", [])

    # fast-reid embedding extractor complexity - shared by every row that passes
    # common_reid_args (+ReID, +DMA, +LTC+DMA all extract embeddings per crop).
    reid_params_m, reid_flops_g = reid_complexity(
        os.path.join(ROOT, cfg["fast_reid_config"]), os.path.join(ROOT, cfg["fast_reid_weights"]),
    )
    reid_note = "ReID: {:.2f}M/{:.2f}G per crop".format(reid_params_m, reid_flops_g)

    # +ReID: fixed-weight IoU/appearance fusion (no DMA), same fast-reid backbone.
    add_row(
        "+ReID",
        common_reid_args + ["--reid-weight", str(cfg["reid_weight"])],
        extra_flops_g=reid_flops_g, extra_params_m=reid_params_m,
        note="fixed reid_weight={}, {}".format(cfg["reid_weight"], reid_note),
    )

    # +DMA: fixed ReID backbone + learned GBM fusion (replaces the fixed weight above).
    gbm_trees, gbm_leaves = gbm_complexity(os.path.join(ROOT, cfg["gbm_weights"]))
    gbm_note = "GBM: {} trees, {} leaves; {}".format(gbm_trees, gbm_leaves, reid_note)
    add_row(
        "+DMA",
        common_reid_args + ["--ml", "gbm", "--ml-weights", os.path.join(ROOT, cfg["gbm_weights"])],
        extra_flops_g=reid_flops_g, extra_params_m=reid_params_m,
        note=gbm_note,
    )

    if cfg["ltc_ckpt"]:
        ltc_params_m, ltc_flops_g = ltc_complexity(os.path.join(ROOT, cfg["ltc_ckpt"]))
        add_row(
            "+LTC",
            ["--ltc-motion-ckpt", os.path.join(ROOT, cfg["ltc_ckpt"])],
            extra_flops_g=ltc_flops_g, extra_params_m=ltc_params_m,
        )
        add_row(
            "+LTC+DMA",
            common_reid_args + [
                "--ml", "gbm", "--ml-weights", os.path.join(ROOT, cfg["gbm_weights"]),
                "--ltc-motion-ckpt", os.path.join(ROOT, cfg["ltc_ckpt"]),
            ],
            extra_flops_g=ltc_flops_g + reid_flops_g, extra_params_m=ltc_params_m + reid_params_m,
            note=gbm_note,
        )
    else:
        print("[warn] no LTC checkpoint configured for dataset '{}' - skipping +LTC / +LTC+DMA rows".format(args.dataset))

    lines = ["| Model | FPS | Latency | FLOPs | Params |", "| --- | --- | --- | --- | --- |"]
    for label, summary, extra_flops_g, extra_params_m, note in rows:
        speed = summary.get("speed") or {}
        complexity = summary.get("detector_complexity") or {}
        fps = speed.get("fps")
        latency_ms = speed.get("total_ms")
        base_flops_g = complexity.get("flops_G", 0.0) + extra_flops_g
        base_params_m = complexity.get("params_M", 0.0) + extra_params_m

        fps_str = "{:.2f}".format(fps) if fps is not None else "N/A"
        latency_str = "{:.2f} ms".format(latency_ms) if latency_ms is not None else "N/A"
        flops_str = fmt_flops(base_flops_g, note)
        params_str = fmt_params(base_params_m, note)

        lines.append("| {} | {} | {} | {} | {} |".format(label, fps_str, latency_str, flops_str, params_str))

    table = "\n".join(lines)
    print("\n" + table)

    md_out = args.md_out or os.path.join(ROOT, args.output_root, "{}_{}_pipeline_benchmark.md".format(args.tag, args.dataset))
    os.makedirs(os.path.dirname(md_out), exist_ok=True)
    with open(md_out, "w") as f:
        f.write(table + "\n")
    print("\nSaved table to {}".format(md_out))


if __name__ == "__main__":
    main()

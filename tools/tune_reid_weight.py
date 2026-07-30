"""Bayesian-optimize `reid_weight` (motion_weight = 1 - reid_weight is implied).

Each trial launches `tools/track_reid.py` as a subprocess with a candidate
`--reid-weight`, then scores the run with HOTA via `tools/track_dma.py:compute_hota`.
Every trial's params + metrics are appended to a CSV as soon as it finishes.

gt_root/gt_type are auto-derived from the exp file's eval dataset (data_dir/name),
same as tools/track_dma.py does -- pass --gt-root/--gt-type explicitly to override.

Example:
    python tools/tune_reid_weight.py \
        -f exps/example/sportmot/yolox_x_sportsmot.py -c pretrained/best_ckpt.pth.tar \
        --n-trials 20 \
        -- -b 1 -d 1 --fp16 --fuse
"""
import argparse
import csv
import os
import subprocess
import sys
import time

import optuna
from optuna.trial import TrialState

FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(FILE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from yolox.exp import get_exp  # noqa: E402
from tools.track_dma import compute_hota  # noqa: E402


def infer_gt_root(exp_file, exp_name):
    """Same auto-derivation track_dma.py:main() does: build the eval dataset once
    (no GPU/model needed, just parses the annotation json) and read its data_dir/name.
    """
    exp = get_exp(exp_file, exp_name)
    val_loader = exp.get_eval_loader(1, False)
    dataset = getattr(val_loader, "dataset", None)
    data_dir = getattr(dataset, "data_dir", None)
    name = getattr(dataset, "name", None)
    if not data_dir or not name:
        return None, None
    gt_root = os.path.join(data_dir, name)
    gt_type = "_val_half" if exp.val_ann == "val_half.json" else ""
    return gt_root, gt_type


def make_parser():
    parser = argparse.ArgumentParser("Tune reid_weight with Bayesian Optimization")
    parser.add_argument("-f", "--exp-file", type=str, required=True, help="forwarded to track_reid.py as -f")
    parser.add_argument("-n", "--name", type=str, default=None, help="model name, forwarded to get_exp")
    parser.add_argument("-c", "--ckpt", type=str, required=True, help="forwarded to track_reid.py as -c")
    parser.add_argument("--gt-root", type=str, default=None,
                         help="MOTChallenge-style GT root; auto-derived from the exp file's eval dataset "
                              "(data_dir/name) if omitted, same as track_dma.py")
    parser.add_argument("--gt-type", type=str, default=None,
                         help="GT suffix, e.g. _val_half -> gt_val_half.txt; auto-derived from exp.val_ann if omitted")
    parser.add_argument("--output-dir", type=str, default="./YOLOX_outputs", help="must match exp.output_dir")
    parser.add_argument("--study-name", type=str, default="tune_reid_weight")
    parser.add_argument("--storage", type=str, default=None,
                         help="Optuna storage URL for resumable studies, e.g. "
                              "sqlite:////abs/path/study.db (default: sqlite file at "
                              "<output-dir>/<study-name>/study.db). Pass 'none' to disable "
                              "persistence and run in-memory (not resumable).")
    parser.add_argument("--low", type=float, default=0.0, help="lower bound for reid_weight")
    parser.add_argument("--high", type=float, default=1.0, help="upper bound for reid_weight")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-startup-trials", type=int, default=5, help="random warm-up trials before GP kicks in")
    parser.add_argument("--patience", type=int, default=0,
                         help="stop early if the best value hasn't improved for this many consecutive "
                              "trials (0 = disabled, always run --n-trials)")
    parser.add_argument("--min-delta", type=float, default=0.0,
                         help="minimum absolute improvement in the metric to reset patience counter")
    parser.add_argument("--metric", type=str, default="HOTA", choices=["HOTA", "DetA", "AssA", "MOTA", "IDF1"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=str, default=None, help="output CSV path (default: <output-dir>/<study-name>/results.csv)")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="everything after `--` is forwarded as-is to tools/track_reid.py "
             "(e.g. -b 1 -d 1 --fp16 --fuse --track_thresh 0.6 ...)",
    )
    return parser


def run_trial(args, extra_args, reid_weight, trial_number):
    expn = f"{args.study_name}/trial_{trial_number:03d}"
    cmd = [
        sys.executable, os.path.join(ROOT, "tools", "track_dma.py"),
        "-f", args.exp_file,
        "-c", args.ckpt,
        "-expn", expn,
        "--with-reid",
        "--reid-weight", str(reid_weight),
    ]
    if args.name:
        cmd += ["-n", args.name]
    cmd += extra_args
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0

    # ReID init logs (e.g. "[ReID] Backend: FastReID | config: ... | weights: ...") are
    # plain print()s from byte_tracker.py/reid.py, not loguru -- they only exist in this
    # subprocess's stdout. Persist them so a successful trial's load can still be verified.
    file_name = os.path.join(args.output_dir, expn)
    os.makedirs(file_name, exist_ok=True)
    log_path = os.path.join(file_name, "tune_stdout.log")
    with open(log_path, "w") as f:
        f.write(proc.stdout)
        f.write("\n----- STDERR -----\n")
        f.write(proc.stderr)

    reid_lines = [ln for ln in proc.stdout.splitlines() if "[ReID]" in ln]
    if reid_lines:
        print(f"trial {trial_number} (reid_weight={reid_weight:.4f}):")
        for ln in reid_lines:
            print(f"  {ln}")
    else:
        print(f"trial {trial_number}: WARNING -- no '[ReID]' line in output, "
              f"with-reid may not have loaded (check {log_path})")

    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
        raise RuntimeError(f"trial {trial_number} failed (reid_weight={reid_weight}): track_dma.py exit {proc.returncode}")

    results_folder = os.path.join(args.output_dir, expn, "track_results")
    metrics = compute_hota(args.gt_root, results_folder, gt_type=args.gt_type)
    if not metrics or metrics.get(args.metric) is None:
        raise RuntimeError(f"trial {trial_number}: HOTA evaluation returned no '{args.metric}' score")
    metrics["elapsed_sec"] = round(elapsed, 1)
    return metrics


def append_csv(csv_path, row, fieldnames):
    is_new = not os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = make_parser().parse_args()
    extra_args = args.extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    if args.gt_root is None or args.gt_type is None:
        inferred_root, inferred_type = infer_gt_root(args.exp_file, args.name)
        if args.gt_root is None:
            if inferred_root is None:
                raise RuntimeError(
                    "Could not auto-derive --gt-root from the exp file's eval dataset; pass --gt-root explicitly."
                )
            args.gt_root = inferred_root
        if args.gt_type is None:
            args.gt_type = inferred_type or ""
        print(f"Auto-derived gt_root={args.gt_root} gt_type={args.gt_type!r}")

    csv_path = args.csv or os.path.join(args.output_dir, args.study_name, "results.csv")
    fieldnames = [
        "trial", "reid_weight", "motion_weight",
        "HOTA", "DetA", "AssA", "MOTA", "MOTP", "IDF1", "IDP", "IDR",
        "elapsed_sec",
    ]

    def objective(trial):
        reid_weight = trial.suggest_float("reid_weight", args.low, args.high)
        metrics = run_trial(args, extra_args, reid_weight, trial.number)

        row = {"trial": trial.number, "reid_weight": reid_weight, "motion_weight": 1.0 - reid_weight}
        row.update(metrics)
        append_csv(csv_path, row, fieldnames)

        for k, v in metrics.items():
            if k != "elapsed_sec":
                trial.set_user_attr(k, v)

        return metrics[args.metric]

    if args.storage is None:
        storage_path = os.path.join(args.output_dir, args.study_name, "study.db")
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        storage = f"sqlite:///{os.path.abspath(storage_path)}"
    elif args.storage.lower() == "none":
        storage = None
    else:
        storage = args.storage

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=optuna.samplers.GPSampler(n_startup_trials=args.n_startup_trials, seed=args.seed),
        storage=storage,
        load_if_exists=True,
    )

    n_done = sum(
        trial.state == TrialState.COMPLETE
        for trial in study.trials
    )
    if n_done > 0:
        print(f"Resuming study '{args.study_name}' from {storage}: {n_done} trial(s) already recorded.")
    n_remaining = max(args.n_trials - n_done, 0)
    if n_remaining == 0:
        print(f"Study already has {n_done} >= --n-trials {args.n_trials}; nothing to do.")
        print(f"Best reid_weight={study.best_params['reid_weight']:.4f} "
              f"(motion_weight={1.0 - study.best_params['reid_weight']:.4f}) "
              f"-> {args.metric}={study.best_value:.3f}")
        return

    callbacks = []
    if args.patience > 0:
        no_improve_count = {"n": 0, "best": None}

        def early_stop_callback(study, trial):
            best = study.best_value
            if no_improve_count["best"] is None or best > no_improve_count["best"] + args.min_delta:
                no_improve_count["best"] = best
                no_improve_count["n"] = 0
            else:
                no_improve_count["n"] += 1
                print(f"  no improvement for {no_improve_count['n']}/{args.patience} trials "
                      f"(best {args.metric}={best:.3f})")
            if no_improve_count["n"] >= args.patience:
                print(f"Early stopping: no improvement in {args.metric} for {args.patience} consecutive trials.")
                study.stop()

        callbacks.append(early_stop_callback)

    study.optimize(objective, n_trials=n_remaining, callbacks=callbacks)

    print(f"\nBest reid_weight={study.best_params['reid_weight']:.4f} "
          f"(motion_weight={1.0 - study.best_params['reid_weight']:.4f}) "
          f"-> {args.metric}={study.best_value:.3f}")
    print(f"Per-trial results saved to {csv_path}")


if __name__ == "__main__":
    main()

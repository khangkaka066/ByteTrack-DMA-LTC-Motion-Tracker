"""
Sweep multiple feature-subset configs through the GBM fused-cost objective
(same one train_gbm.py / analyze_features.py use) and rank them by val_f1.

Each config is a named subset of features.py's FEAT_NAMES columns - no need
to re-run generate_data.py, as long as the subset only drops columns (never
adds new ones the stored .npz doesn't have).

Usage:
  python -m yolox.DMA.sweep_gbm \\
    --data-dir datasets/mot17_dma --val-data-dir datasets/mot17_dma_valhalf \\
    --config baseline=all \\
    --config no_cosine=drop:cosine_dist \\
    --config motion_only=motion_cost,mahalanobis_norm,cov_trace_log \\
    --out dma_weights/gbm_sweep_mot17.json

Leave-one-out over the current feature set (one config per column, e.g.
'drop_cosine_dist') without typing anything by hand:
  python -m yolox.DMA.sweep_gbm --data-dir ... --val-data-dir ... --loo

Or load configs from a JSON file (name -> "all" | list[int] | list[str] | "drop:...")
  {"baseline": "all", "no_cosine": "drop:cosine_dist"}

  python -m yolox.DMA.sweep_gbm --data-dir ... --val-data-dir ... \\
    --configs-file configs/gbm_sweep.json

--config / --configs-file / --loo are all optional and combine (repeat
--config, or mix with --loo) - at least one source of configs is required.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from yolox.DMA.dataset import DMADataset
from yolox.DMA.model_gbm import DynamicWeightGBM
from yolox.DMA.gbm_objective import make_bce_fused_objective, make_fused_eval
from yolox.DMA.features import FEAT_DIM, FEAT_NAMES
from yolox.DMA.feature_spec import resolve_feature_spec, loo_configs


def parse_config_arg(s: str):
    """'name=<spec>' -> (name, indices | None). See resolve_feature_spec for <spec> syntax."""
    name, _, spec = s.partition("=")
    return name, resolve_feature_spec(spec)


def load_configs(args) -> "dict[str, list | None]":
    configs = {}
    if args.configs_file:
        with open(args.configs_file) as f:
            raw = json.load(f)
        for name, spec in raw.items():
            configs[name] = resolve_feature_spec(spec if isinstance(spec, str) else ",".join(map(str, spec)))
    for c in args.config or []:
        name, idx = parse_config_arg(c)
        configs[name] = idx
    if args.loo:
        configs.setdefault("baseline", None)
        configs.update(loo_configs())
    if not configs:
        raise ValueError(
            "No configs given: use --config name=<spec> (repeatable), --configs-file, and/or --loo"
        )
    return configs


def train_one(train_paths, val_paths, feature_indices, args):
    if val_paths:
        train_ds = DMADataset(
            train_paths, normalize=True, pos_neg_ratio=args.pos_neg_ratio,
            feature_indices=feature_indices,
        )
        val_ds = DMADataset(val_paths, normalize=False, feature_indices=feature_indices)
        val_ds.features = (val_ds.features - train_ds.mean) / train_ds.std
    else:
        train_ds, val_ds = DMADataset.split(
            train_paths, val_ratio=args.val_ratio, normalize=True,
            pos_neg_ratio=args.pos_neg_ratio, seed=args.split_seed,
            feature_indices=feature_indices,
        )

    train_set = lgb.Dataset(train_ds.features, label=train_ds.labels, free_raw_data=False)
    val_set = lgb.Dataset(val_ds.features, label=val_ds.labels, reference=train_set, free_raw_data=False)

    objective = make_bce_fused_objective(train_ds.motion_costs, train_ds.appearance_costs, train_ds.labels)
    val_feval = make_fused_eval(val_ds.motion_costs, val_ds.appearance_costs, val_ds.labels, name="val_f1")

    params = {
        "boosting_type": "gbdt",
        "objective": objective,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "learning_rate": args.lr,
        "min_child_samples": args.min_child_samples,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "boost_from_average": False,
        "verbosity": -1,
    }
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[val_set],
        feval=val_feval,
        callbacks=[lgb.early_stopping(args.early_stopping, verbose=False), lgb.log_evaluation(0)],
    )
    best_f1 = booster.best_score["valid_0"]["val_f1"]
    return booster, train_ds, val_ds, best_f1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--val-data-dir", default=None,
                         help="If omitted, --val-ratio of --data-dir sequence files is held out per config.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--config", action="append",
                         help="'name=idx1,idx2,...' or 'name=all'. Repeatable.")
    parser.add_argument("--configs-file", default=None, help="JSON file: {name: 'all' | [indices] | 'drop:name,...'}")
    parser.add_argument("--loo", action="store_true",
                         help="Auto-add a leave-one-out config per current feature "
                              "('drop_<name>') plus 'baseline'=all, on top of any --config/--configs-file.")
    parser.add_argument("--num-boost-round", type=int, default=50)
    parser.add_argument("--early-stopping", type=int, default=30)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--pos-neg-ratio", type=float, default=5.0)
    parser.add_argument("--out", default=None, help="Optional path to dump full JSON report")
    parser.add_argument("--save-checkpoints-dir", default=None,
                         help="If set, save each config's booster as <dir>/<name>.gbm")
    args = parser.parse_args()

    configs = load_configs(args)

    train_paths = sorted(str(p) for p in Path(args.data_dir).glob("*.npz"))
    if not train_paths:
        raise FileNotFoundError(f"No .npz files found in {args.data_dir}")
    val_paths = None
    if args.val_data_dir:
        val_paths = sorted(str(p) for p in Path(args.val_data_dir).glob("*.npz"))
        if not val_paths:
            raise FileNotFoundError(f"No .npz files found in {args.val_data_dir}")

    results = []
    for name, feature_indices in configs.items():
        feat_desc = f"all ({FEAT_DIM})" if feature_indices is None else str(feature_indices)
        print(f"\n{'='*70}\n[{name}] feature_indices={feat_desc}\n{'='*70}")
        t0 = time.time()
        booster, train_ds, val_ds, best_f1 = train_one(train_paths, val_paths, feature_indices, args)
        elapsed = time.time() - t0
        used_names = FEAT_NAMES if feature_indices is None else [FEAT_NAMES[i] for i in feature_indices]
        print(f"  n_train={len(train_ds.labels):,}  n_val={len(val_ds.labels):,}")
        print(f"  best_iteration={booster.best_iteration}  best_val_f1={best_f1:.4f}  ({elapsed:.1f}s)")

        if args.save_checkpoints_dir:
            out_dir = Path(args.save_checkpoints_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            truncated = lgb.Booster(model_str=booster.model_to_string(num_iteration=booster.best_iteration))
            stats = {
                "mean": train_ds.mean.tolist(),
                "std": train_ds.std.tolist(),
                "feature_indices": feature_indices if feature_indices is not None else list(range(FEAT_DIM)),
            }
            DynamicWeightGBM(truncated).save(str(out_dir / f"{name}.gbm"), stats=stats)

        results.append({
            "config": name,
            "feature_indices": feature_indices if feature_indices is not None else list(range(FEAT_DIM)),
            "feature_names": used_names,
            "n_train": int(len(train_ds.labels)),
            "n_val": int(len(val_ds.labels)),
            "best_iteration": int(booster.best_iteration),
            "best_val_f1": float(best_f1),
            "elapsed_sec": round(elapsed, 1),
        })

    results.sort(key=lambda r: -r["best_val_f1"])
    print(f"\n{'='*70}\nSUMMARY (ranked by val_f1)\n{'='*70}")
    print(f"  {'config':<20} {'n_feat':>6} {'best_val_f1':>12} {'best_iter':>10}")
    for r in results:
        print(f"  {r['config']:<20} {len(r['feature_indices']):>6} {r['best_val_f1']:>12.4f} {r['best_iteration']:>10}")
    print("\nNote: val_f1 is a pair-level proxy (fused-cost BCE objective), not a")
    print("tracking-level metric. Confirm the top 2-3 configs with a real HOTA/IDF1")
    print("ablation (idea_DMA.md section 14) before finalizing features.py.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()

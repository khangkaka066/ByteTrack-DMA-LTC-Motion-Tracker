"""
Feature selection analysis for DynamicWeightNet's feature vector
(see yolox/DMA/features.py FEAT_NAMES).

Answers "which of the current features should I keep or drop?" from three angles:

  1. GBM feature importance (gain + split) - trains a fresh LightGBM booster
     with the same fused-cost BCE objective used in train_gbm.py, then reads
     off per-feature contribution to the actual objective (not a generic
     classification proxy).
  2. Correlation matrix - flags redundant feature pairs (e.g. motion_cost vs
     mahalanobis_norm, which move together; see idea_DMA.md 5.6).
  3. Permutation importance - shuffles one feature column at a time on the
     held-out val set and measures the drop in val_f1 (the same metric
     train_gbm.py early-stops on). This is the closest proxy we have to
     "does removing this feature actually hurt matching quality" without
     re-running full tracking + HOTA evaluation.

None of this replaces the tracking-level ablation in idea_DMA.md section 14
(Motion only / no covariance / no ReID / etc. re-evaluated with HOTA/IDF1 on
a real sequence) - that is still the final word. This script is the cheap,
fast filter to decide which ablations are worth running.

Usage:
  python -m yolox.DMA.analyze_features \\
    --data-dir     datasets/mot17_dma \\
    --val-data-dir datasets/mot17_dma_val \\
    --out          dma_weights/feature_analysis_mot17.json

If you don't have a separate validation split, omit --val-data-dir: the
script holds out --val-fraction of the *sequence files* in --data-dir
(via DMADataset.split, same file-level split used by train.py/train_gbm.py)
instead of using them for training. With very few .npz files this holdout
is noisy (e.g. 1 sequence out of 3) - prefer a real --val-data-dir when
you have one.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from yolox.DMA.dataset import DMADataset
from yolox.DMA.gbm_objective import make_bce_fused_objective, make_fused_eval
from yolox.DMA.features import FEAT_NAMES


def compute_correlation(features: np.ndarray, threshold: float = 0.9):
    corr = np.corrcoef(features, rowvar=False)
    pairs = []
    n = corr.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) >= threshold:
                pairs.append({
                    "feature_a": FEAT_NAMES[i],
                    "feature_b": FEAT_NAMES[j],
                    "corr": float(corr[i, j]),
                })
    pairs.sort(key=lambda p: -abs(p["corr"]))
    return corr, pairs


def train_reference_booster(train_ds, val_ds, args):
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
    return booster


def feature_importance(booster: lgb.Booster):
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    gain_norm = gain / (gain.sum() + 1e-8)
    order = np.argsort(-gain_norm)
    return [
        {
            "feature": FEAT_NAMES[i],
            "gain": float(gain[i]),
            "gain_frac": float(gain_norm[i]),
            "split_count": int(split[i]),
        }
        for i in order
    ]


def permutation_importance(booster: lgb.Booster, val_ds, n_repeats: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    feval = make_fused_eval(val_ds.motion_costs, val_ds.appearance_costs, val_ds.labels, name="val_f1")

    def val_f1(feats):
        z = booster.predict(feats, raw_score=True)
        _, score, _ = feval(z, None)
        return score

    baseline = val_f1(val_ds.features)
    results = []
    for i, name in enumerate(FEAT_NAMES):
        drops = []
        for _ in range(n_repeats):
            perturbed = val_ds.features.copy()
            rng.shuffle(perturbed[:, i])
            drops.append(baseline - val_f1(perturbed))
        results.append({
            "feature": name,
            "baseline_val_f1": float(baseline),
            "mean_f1_drop": float(np.mean(drops)),
            "std_f1_drop": float(np.std(drops)),
        })
    results.sort(key=lambda r: -r["mean_f1_drop"])
    return baseline, results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="Dir of training .npz files (see generate_data.py)")
    parser.add_argument("--val-data-dir", default=None,
                         help="Dir of validation .npz files. If omitted, --val-fraction of the "
                              "sequence files in --data-dir is held out instead.")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                         help="Fraction of --data-dir files held out as val when --val-data-dir is omitted")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Optional path to dump full JSON report")
    parser.add_argument("--corr-threshold", type=float, default=0.9)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=30)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--n-repeats", type=int, default=5, help="Repeats per feature for permutation importance")
    args = parser.parse_args()

    train_paths = sorted(Path(args.data_dir).glob("*.npz"))
    if not train_paths:
        raise FileNotFoundError(f"No .npz files found in {args.data_dir}")

    # normalize=False: importance/correlation/permutation should read on the
    # raw feature scale features.py actually emits, not a z-scored copy.
    if args.val_data_dir:
        val_paths = sorted(Path(args.val_data_dir).glob("*.npz"))
        if not val_paths:
            raise FileNotFoundError(f"No .npz files found in {args.val_data_dir}")
        train_ds = DMADataset([str(p) for p in train_paths], normalize=False)
        val_ds = DMADataset([str(p) for p in val_paths], normalize=False)
    else:
        print(f"No --val-data-dir given: holding out {args.val_fraction:.0%} of the "
              f"{len(train_paths)} sequence files in {args.data_dir} for validation.")
        train_ds, val_ds = DMADataset.split(
            [str(p) for p in train_paths],
            val_ratio=args.val_fraction,
            normalize=False,
            seed=args.split_seed,
        )
    print(f"Train pairs: {len(train_ds.labels)}  |  Val pairs: {len(val_ds.labels)}")

    print("\n[1/3] Training reference LightGBM booster (fused-cost BCE objective)...")
    booster = train_reference_booster(train_ds, val_ds, args)

    print("\n[2/3] Feature importance (gain / split):")
    importances = feature_importance(booster)
    for r in importances:
        print(f"  {r['feature']:<20s} gain_frac={r['gain_frac']:.4f}  splits={r['split_count']}")

    print(f"\n[3/3] Correlation pairs with |corr| >= {args.corr_threshold}:")
    _, corr_pairs = compute_correlation(train_ds.features, threshold=args.corr_threshold)
    if not corr_pairs:
        print("  (none)")
    for p in corr_pairs:
        print(f"  {p['feature_a']:<20s} <-> {p['feature_b']:<20s}  corr={p['corr']:.3f}")

    print("\nPermutation importance on val set (val_f1 drop when feature is shuffled):")
    baseline_f1, perm_results = permutation_importance(booster, val_ds, n_repeats=args.n_repeats)
    print(f"  baseline val_f1 = {baseline_f1:.4f}")
    for r in perm_results:
        print(f"  {r['feature']:<20s} mean_drop={r['mean_f1_drop']:+.4f}  std={r['std_f1_drop']:.4f}")

    low_gain = {r["feature"] for r in importances if r["gain_frac"] < 0.01}
    low_perm = {r["feature"] for r in perm_results if r["mean_f1_drop"] < 0.002}
    drop_candidates = sorted(low_gain & low_perm)
    print("\nCandidates to drop (low GBM gain AND low permutation impact):")
    print(f"  {drop_candidates if drop_candidates else '(none)'}")
    print("Note: does not include the mandatory idx-14 'has_appearance' flag semantics check -")
    print("      a low-importance flag can still be structurally required (see idea_DMA.md 5.2/5.6).")
    print("Final decision should still be confirmed with a tracking-level HOTA/IDF1 ablation")
    print("(idea_DMA.md section 14) before removing anything from features.py.")

    if args.out:
        report = {
            "n_train": int(len(train_ds.labels)),
            "n_val": int(len(val_ds.labels)),
            "feature_importance": importances,
            "correlation_pairs": corr_pairs,
            "permutation_importance": {"baseline_val_f1": float(baseline_f1), "results": perm_results},
            "drop_candidates": drop_candidates,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()

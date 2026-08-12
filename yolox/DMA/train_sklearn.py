"""
Train a scikit-learn RandomForest or LogisticRegression as a DMA
fusion-weight predictor.

Uses the same .npz data pipeline (DMADataset) as train.py / train_gbm.py /
train_xgb.py, but a different training signal: see model_sklearn.py for why
(no generic custom-objective hook in sklearn).
  rf      MSE regression onto a closed-form per-sample target weight.
  logreg  Classification on the binarised target weight; predict_proba
          gives a continuous [0, 1] w_motion at inference.
Treat comparisons against MLP/GBM/XGB as "does the predictor family
matter" rather than a perfectly controlled loss-for-loss ablation.

Usage:
  python -m yolox.DMA.train_sklearn \\
    --data-dir data/dma_train \\
    --out-dir  weights/dma_sklearn_rf \\
    --algo     rf
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from yolox.DMA.dataset import DMADataset
from yolox.DMA.model_sklearn import DynamicWeightSklearn, soft_weight_target, predict_w_motion
from yolox.DMA.features import FEAT_DIM
from yolox.DMA.feature_spec import resolve_feature_spec


def build_estimator(algo: str, args):
    if algo == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth > 0 else None,
            min_samples_leaf=args.min_child_samples,
            n_jobs=-1, random_state=42,
        )
    if algo == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=args.C, max_iter=args.max_iter, n_jobs=-1)
    raise ValueError(f"Unknown --algo {algo}")


def fit_target(algo: str, motion_cost, appearance_cost, labels):
    soft = soft_weight_target(motion_cost, appearance_cost, labels)
    if algo == "logreg":
        return (soft >= 0.5).astype(np.int64)  # "should motion dominate for this pair"
    return soft


def evaluate(estimator, val_ds) -> float:
    """F1 on the fused-cost < 0.5 decision, matching evaluate() in train.py."""
    w_motion = predict_w_motion(estimator, val_ds.features)
    fused = w_motion * val_ds.motion_costs + (1.0 - w_motion) * val_ds.appearance_costs
    pred_match = (fused < 0.5).astype(np.float64)
    lbl = val_ds.labels.astype(np.float64)

    tp = np.sum((pred_match == 1) & (lbl == 1))
    fp = np.sum((pred_match == 1) & (lbl == 0))
    fn = np.sum((pred_match == 0) & (lbl == 1))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(2 * precision * recall / (precision + recall + 1e-8))


def train(args):
    data_paths = sorted(Path(args.data_dir).glob("*.npz"))
    if not data_paths:
        raise FileNotFoundError(f"No .npz files found in {args.data_dir}")
    print(f"Found {len(data_paths)} train sequences")

    feature_indices = resolve_feature_spec(args.feature_indices)
    if feature_indices:
        print(f"Using feature subset (indices={feature_indices})")

    if args.val_data_dir:
        val_paths = sorted(Path(args.val_data_dir).glob("*.npz"))
        if not val_paths:
            raise FileNotFoundError(f"No .npz files found in {args.val_data_dir}")
        print(f"Found {len(val_paths)} val sequences")
        train_ds = DMADataset(
            [str(p) for p in data_paths], normalize=True, pos_neg_ratio=args.pos_neg_ratio,
            feature_indices=feature_indices,
        )
        val_ds = DMADataset([str(p) for p in val_paths], normalize=False, feature_indices=feature_indices)
        val_ds.features = (val_ds.features - train_ds.mean) / train_ds.std
        val_ds.mean = train_ds.mean
        val_ds.std = train_ds.std
    else:
        train_ds, val_ds = DMADataset.split(
            [str(p) for p in data_paths],
            val_ratio=args.val_ratio, normalize=True, pos_neg_ratio=args.pos_neg_ratio,
            feature_indices=feature_indices,
        )

    pos, neg = train_ds.class_balance()
    print(f"Train: {len(train_ds)} samples  pos={pos}  neg={neg}  ratio=1:{neg // max(pos, 1)}")
    print(f"Val:   {len(val_ds)} samples")

    if train_ds.motion_costs is None or train_ds.appearance_costs is None:
        raise ValueError(
            "motion_costs/appearance_costs missing from .npz data — "
            "required to build the soft-weight training target."
        )

    target = fit_target(args.algo, train_ds.motion_costs, train_ds.appearance_costs, train_ds.labels)
    estimator = build_estimator(args.algo, args)

    print(f"Fitting {args.algo} on {train_ds.features.shape[0]} samples...")
    estimator.fit(train_ds.features, target)

    val_f1 = evaluate(estimator, val_ds)
    print(f"Val fused-cost F1: {val_f1:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "mean": train_ds.mean.tolist(),
        "std": train_ds.std.tolist(),
        "feature_indices": feature_indices if feature_indices is not None else list(range(FEAT_DIM)),
    }
    ckpt = str(out_dir / f"dma_sklearn_{args.algo}.skl")
    DynamicWeightSklearn(estimator, args.algo).save(ckpt, stats=stats)
    with open(out_dir / "normalization_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved checkpoint: {ckpt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--val-data-dir", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--algo", required=True, choices=["rf", "logreg"])
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--pos-neg-ratio", type=float, default=5.0)
    # rf
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=-1, help="<=0 means unlimited")
    parser.add_argument("--min-child-samples", type=int, default=20, help="min_samples_leaf")
    # logreg
    parser.add_argument("--C", type=float, default=1.0, help="LogisticRegression inverse regularisation")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--feature-indices", default=None,
                         help="Ablate features.py's FEAT_NAMES columns without re-running "
                              "generate_data.py. Accepts indices ('0,2,3'), names "
                              "('cosine_dist,tracklet_len_norm'), or 'drop:name1,name2' to keep "
                              "all but those. Omit to use all features (default).")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

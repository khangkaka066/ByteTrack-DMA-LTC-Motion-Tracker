"""
Train DynamicWeightXGB (XGBoost alternative to DynamicWeightNet / DynamicWeightGBM).

Uses the exact same .npz data pipeline (DMADataset) and the exact same
fused-cost BCE custom objective as train_gbm.py (see gbm_objective.py), so
results are directly comparable across MLP / LightGBM / XGBoost.

Usage:
  python -m yolox.DMA.train_xgb \\
    --data-dir  data/dma_train \\
    --out-dir   weights/dma_xgb \\
    --num-boost-round 500
"""

import argparse
import json
import sys
from pathlib import Path

import xgboost as xgb

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from yolox.DMA.dataset import DMADataset
from yolox.DMA.model_xgb import DynamicWeightXGB
from yolox.DMA.gbm_objective import make_bce_fused_objective, make_fused_eval
from yolox.DMA.features import FEAT_DIM
from yolox.DMA.feature_spec import resolve_feature_spec


def _as_xgb_feval(lgb_style_feval):
    """Adapt a (preds, data) -> (name, value, is_higher_better) feval (LightGBM
    convention, used by gbm_objective.make_fused_eval) to xgboost's
    custom_metric convention: (preds, dmatrix) -> (name, value)."""

    def feval(preds, dmatrix):
        name, value, _ = lgb_style_feval(preds, dmatrix)
        return name, value

    return feval


def train(args):
    if args.loss != "bce":
        raise NotImplementedError(
            "train_xgb.py only implements the 'bce' fused-cost objective "
            "(see gbm_objective.py). RankingLoss has no custom-objective "
            "equivalent here; use train.py --loss ranking for the MLP instead."
        )

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
            val_ratio=args.val_ratio,
            normalize=True,
            pos_neg_ratio=args.pos_neg_ratio,
            feature_indices=feature_indices,
        )

    pos, neg = train_ds.class_balance()
    print(f"Train: {len(train_ds)} samples  pos={pos}  neg={neg}  ratio=1:{neg // max(pos, 1)}")
    print(f"Val:   {len(val_ds)} samples")

    if train_ds.motion_costs is None or train_ds.appearance_costs is None:
        raise ValueError(
            "motion_costs/appearance_costs missing from .npz data — "
            "required to build the fused-cost objective."
        )

    dtrain = xgb.DMatrix(train_ds.features, label=train_ds.labels)
    dval = xgb.DMatrix(val_ds.features, label=val_ds.labels)

    train_objective = make_bce_fused_objective(
        train_ds.motion_costs, train_ds.appearance_costs, train_ds.labels
    )
    val_feval = _as_xgb_feval(
        make_fused_eval(val_ds.motion_costs, val_ds.appearance_costs, val_ds.labels, name="val_f1")
    )

    params = {
        "booster": "gbtree",
        "max_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "grow_policy": "lossguide" if args.num_leaves > 0 else "depthwise",
        "learning_rate": args.lr,
        "min_child_weight": args.min_child_samples,
        "colsample_bytree": 0.9,
        "subsample": 0.9,
        "base_score": 0.0,   # custom objective: start margins at 0 (like boost_from_average=False)
        "verbosity": 0,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evals_result = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_boost_round,
        evals=[(dval, "valid_0")],
        obj=train_objective,
        custom_metric=val_feval,
        maximize=True,
        early_stopping_rounds=args.early_stopping,
        verbose_eval=args.eval_every,
        evals_result=evals_result,
    )

    stats = {
        "mean": train_ds.mean.tolist(),
        "std": train_ds.std.tolist(),
        "feature_indices": feature_indices if feature_indices is not None else list(range(FEAT_DIM)),
    }

    best_ckpt = str(out_dir / "dma_xgb_best.xgb")
    best_iteration = booster.best_iteration
    truncated = booster[: best_iteration + 1]
    DynamicWeightXGB(truncated).save(best_ckpt, stats=stats)

    with open(out_dir / "normalization_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    best_val_f1 = evals_result["valid_0"]["val_f1"][best_iteration]
    print(f"\nTraining complete. Best iteration: {best_iteration}")
    print(f"Best val_f1: {best_val_f1:.4f}")
    print(f"Best checkpoint: {best_ckpt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--val-data-dir", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--loss", choices=["bce", "ranking"], default="bce")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--pos-neg-ratio", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--feature-indices", default=None,
                         help="Ablate features.py's FEAT_NAMES columns without re-running "
                              "generate_data.py. Accepts indices ('0,2,3'), names "
                              "('cosine_dist,tracklet_len_norm'), or 'drop:name1,name2' to keep "
                              "all but those. Omit to use all features (default).")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

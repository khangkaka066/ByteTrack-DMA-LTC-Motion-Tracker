"""
Train DynamicWeightGBM (LightGBM alternative to DynamicWeightNet).

Uses the exact same .npz data pipeline (DMADataset) and the exact same
fused-cost BCE loss as train.py's --loss bce, implemented as a LightGBM
custom objective (see gbm_objective.py) so results are directly comparable
to the MLP.

Usage:
  python -m yolox.DMA.train_gbm \\
    --data-dir  data/dma_train \\
    --out-dir   weights/dma_gbm \\
    --num-boost-round 500
"""

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from yolox.DMA.dataset import DMADataset
from yolox.DMA.model_gbm import DynamicWeightGBM
from yolox.DMA.gbm_objective import make_bce_fused_objective, make_fused_eval
from yolox.DMA.features import FEAT_DIM
from yolox.DMA.feature_spec import resolve_feature_spec


def train(args):
    if args.loss != "bce":
        raise NotImplementedError(
            "train_gbm.py only implements the 'bce' fused-cost objective "
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

    train_set = lgb.Dataset(train_ds.features, label=train_ds.labels, free_raw_data=False)
    val_set = lgb.Dataset(
        val_ds.features, label=val_ds.labels, reference=train_set, free_raw_data=False
    )

    train_objective = make_bce_fused_objective(
        train_ds.motion_costs, train_ds.appearance_costs, train_ds.labels
    )
    val_feval = make_fused_eval(
        val_ds.motion_costs, val_ds.appearance_costs, val_ds.labels, name="val_f1"
    )

    params = {
        "boosting_type": "gbdt",
        "objective": train_objective,  # custom fused-cost BCE objective (LightGBM >=4.0 API)
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "learning_rate": args.lr,
        "min_child_samples": args.min_child_samples,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "boost_from_average": False,   # custom objective: start margins at 0
        "verbosity": -1,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[val_set],
        feval=val_feval,
        callbacks=[
            lgb.early_stopping(args.early_stopping),
            lgb.log_evaluation(args.eval_every),
        ],
    )

    stats = {
        "mean": train_ds.mean.tolist(),
        "std": train_ds.std.tolist(),
        "feature_indices": feature_indices if feature_indices is not None else list(range(FEAT_DIM)),
    }

    best_ckpt = str(out_dir / "dma_gbm_best.gbm")
    DynamicWeightGBM(booster).save(best_ckpt, stats=stats)
    # Re-save truncated to the best iteration found by early stopping
    truncated = lgb.Booster(model_str=booster.model_to_string(num_iteration=booster.best_iteration))
    DynamicWeightGBM(truncated).save(best_ckpt, stats=stats)

    with open(out_dir / "normalization_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nTraining complete. Best iteration: {booster.best_iteration}")
    print(f"Best val_f1: {booster.best_score['valid_0']['val_f1']:.4f}")
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
    parser.add_argument("--max-depth", type=int, default=-1)
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

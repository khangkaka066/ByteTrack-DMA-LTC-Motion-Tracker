"""
Bayesian hyperparameter search (Optuna, TPE) for DynamicWeightGBM.

Uses the exact same data pipeline and fused-cost BCE objective as
train_gbm.py — only the LightGBM hyperparameters are searched, the
loss/target formulation stays identical so results stay comparable.

Usage:
  python -m yolox.DMA.tune_gbm \\
    --data-dir  datasets/mot17_dma \\
    --out-dir   weights/dma_gbm_mot17 \\
    --n-trials  50

--feature-indices restricts the search to a subset of features.py's
FEAT_NAMES columns (indices, names, or 'drop:name1,name2' - see
yolox/DMA/feature_spec.py), e.g. to tune a leave-one-out config:
  python -m yolox.DMA.tune_gbm \\
    --data-dir datasets/mot17_dma_train --val-data-dir datasets/mot17_dma_val \\
    --out-dir weights/dma_gbm_mot17_drop_cosine --feature-indices drop:cosine_dist
"""

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import optuna

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from yolox.DMA.dataset import DMADataset
from yolox.DMA.model_gbm import DynamicWeightGBM
from yolox.DMA.gbm_objective import make_bce_fused_objective, make_fused_eval
from yolox.DMA.features import FEAT_DIM
from yolox.DMA.feature_spec import resolve_feature_spec


def _suggest_params(trial: optuna.Trial, gpu_params: dict) -> dict:
    return {
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 500),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-6, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-6, 10.0, log=True),
        "boost_from_average": False,   # custom objective: start margins at 0
        "feature_pre_filter": False,   # min_child_samples varies per trial; disable the
                                        # Dataset-level cache that assumes a fixed value
        "verbosity": -1,
        "num_threads": 5,
        **gpu_params,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--val-data-dir", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=None, help="seconds, optional wall-clock budget")
    parser.add_argument("--num-boost-round", type=int, default=6000, help="cap per trial; early stopping usually triggers first")
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--pos-neg-ratio", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-indices", default=None,
                         help="Ablate features.py's FEAT_NAMES columns without re-running "
                              "generate_data.py. Accepts indices ('0,2,3'), names "
                              "('cosine_dist,tracklet_len_norm'), or 'drop:name1,name2' to keep "
                              "all but those. Omit to use all features (default).")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu", "cuda"],
                         help="'gpu' uses LightGBM's OpenCL backend (works with the stock PyPI "
                              "wheel); 'cuda' needs a from-source build with -DUSE_CUDA=1.")
    parser.add_argument("--gpu-platform-id", type=int, default=-1)
    parser.add_argument("--gpu-device-id", type=int, default=-1)
    args = parser.parse_args()

    gpu_params = {}
    if args.device != "cpu":
        gpu_params = {
            "device_type": args.device,
            "gpu_platform_id": args.gpu_platform_id,
            "gpu_device_id": args.gpu_device_id,
        }

    feature_indices = resolve_feature_spec(args.feature_indices)
    if feature_indices:
        print(f"Using feature subset (indices={feature_indices}), input_dim={len(feature_indices)}")

    data_paths = sorted(Path(args.data_dir).glob("*.npz"))
    if not data_paths:
        raise FileNotFoundError(f"No .npz files found in {args.data_dir}")

    if args.val_data_dir:
        val_paths = sorted(Path(args.val_data_dir).glob("*.npz"))
        train_ds = DMADataset(
            [str(p) for p in data_paths], normalize=True, pos_neg_ratio=args.pos_neg_ratio,
            feature_indices=feature_indices,
        )
        val_ds = DMADataset([str(p) for p in val_paths], normalize=False, feature_indices=feature_indices)
        val_ds.features = (val_ds.features - train_ds.mean) / train_ds.std
        val_ds.mean, val_ds.std = train_ds.mean, train_ds.std
    else:
        train_ds, val_ds = DMADataset.split(
            [str(p) for p in data_paths], val_ratio=args.val_ratio,
            normalize=True, pos_neg_ratio=args.pos_neg_ratio, seed=args.seed,
            feature_indices=feature_indices,
        )
    print(f"Train: {len(train_ds)} samples  Val: {len(val_ds)} samples")

    train_set = lgb.Dataset(train_ds.features, label=train_ds.labels, free_raw_data=False)
    val_set = lgb.Dataset(val_ds.features, label=val_ds.labels, reference=train_set, free_raw_data=False)

    train_objective = make_bce_fused_objective(train_ds.motion_costs, train_ds.appearance_costs, train_ds.labels)
    val_feval = make_fused_eval(val_ds.motion_costs, val_ds.appearance_costs, val_ds.labels, name="val_f1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # create up front so progress is visible immediately

    def _save_progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        # Persist best-so-far after every trial so a crash/interrupt doesn't lose all progress
        with open(out_dir / "best_gbm_params.json", "w") as f:
            json.dump({
                "val_f1": study.best_value,
                "params": study.best_params,
                "trials_completed": len(study.trials),
            }, f, indent=2)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, gpu_params)
        params["objective"] = train_objective
        booster = lgb.train(
            params, train_set,
            num_boost_round=args.num_boost_round,
            valid_sets=[val_set],
            feval=val_feval,
            callbacks=[lgb.early_stopping(args.early_stopping, verbose=False)],
        )
        trial.set_user_attr("best_iteration", booster.best_iteration)
        return booster.best_score["valid_0"]["val_f1"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, callbacks=[_save_progress_callback])

    print(f"\nBest val_f1: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")

    # Retrain a final model with the best params (generous round budget, same early stopping)
    final_params = dict(study.best_params)
    final_params.update({"boosting_type": "gbdt", "objective": train_objective,
                          "boost_from_average": False, "feature_pre_filter": False,
                          "verbosity": -1, **gpu_params})
    booster = lgb.train(
        final_params, train_set,
        num_boost_round=max(args.num_boost_round, study.best_trial.user_attrs["best_iteration"] + args.early_stopping),
        valid_sets=[val_set], feval=val_feval,
        callbacks=[lgb.early_stopping(args.early_stopping), lgb.log_evaluation(25)],
    )

    stats = {
        "mean": train_ds.mean.tolist(),
        "std": train_ds.std.tolist(),
        "feature_indices": feature_indices if feature_indices is not None else list(range(FEAT_DIM)),
    }
    best_ckpt = str(out_dir / "dma_gbm_tuned.gbm")
    truncated = lgb.Booster(model_str=booster.model_to_string(num_iteration=booster.best_iteration))
    DynamicWeightGBM(truncated).save(best_ckpt, stats=stats)
    print(f"Final tuned checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()

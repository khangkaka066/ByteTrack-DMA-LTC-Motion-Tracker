"""
Bayesian hyperparameter search (Optuna, TPE) for DynamicWeightXGB.

Uses the exact same data pipeline and fused-cost BCE objective as
train_xgb.py — only the XGBoost hyperparameters are searched, the
loss/target formulation stays identical so results stay comparable with
tune_gbm.py.

Usage:
  python -m yolox.DMA.tune_xgb \\
    --data-dir  datasets/mot17_dma \\
    --out-dir   weights/dma_xgb_mot17 \\
    --n-trials  50
"""

import argparse
import json
import sys
from pathlib import Path

import optuna
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


def _suggest_params(trial: optuna.Trial) -> dict:
    num_leaves = trial.suggest_int("num_leaves", 15, 127)
    return {
        "booster": "gbtree",
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
        "num_leaves": num_leaves,
        "max_leaves": num_leaves,
        "grow_policy": "lossguide",
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500),
        "min_child_weight": trial.suggest_int("min_child_samples", 20, 500),
        "colsample_bytree": trial.suggest_float("feature_fraction", 0.7, 1.0),
        "subsample": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "base_score": 0.0,   # custom objective: start margins at 0
        "verbosity": 0,
        "nthread": 15,
    }


def _params_for_xgb(raw_params: dict) -> dict:
    """Strip Optuna/LightGBM-named duplicate keys, keep only xgboost-native ones."""
    return {
        "booster": "gbtree",
        "learning_rate": raw_params["learning_rate"],
        "max_leaves": raw_params["num_leaves"],
        "grow_policy": "lossguide",
        "max_depth": raw_params["max_depth"],
        "min_child_weight": raw_params["min_child_samples"],
        "colsample_bytree": raw_params["feature_fraction"],
        "subsample": raw_params["bagging_fraction"],
        "reg_alpha": raw_params["lambda_l1"],
        "reg_lambda": raw_params["lambda_l2"],
        "base_score": 0.0,
        "verbosity": 0,
        "nthread": 15,
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
    args = parser.parse_args()

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

    dtrain = xgb.DMatrix(train_ds.features, label=train_ds.labels)
    dval = xgb.DMatrix(val_ds.features, label=val_ds.labels)

    train_objective = make_bce_fused_objective(train_ds.motion_costs, train_ds.appearance_costs, train_ds.labels)
    val_feval = _as_xgb_feval(
        make_fused_eval(val_ds.motion_costs, val_ds.appearance_costs, val_ds.labels, name="val_f1")
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # create up front so progress is visible immediately

    def _save_progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        # Persist best-so-far after every trial so a crash/interrupt doesn't lose all progress
        with open(out_dir / "best_xgb_params.json", "w") as f:
            json.dump({
                "val_f1": study.best_value,
                "params": study.best_params,
                "trials_completed": len(study.trials),
            }, f, indent=2)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        evals_result = {}
        booster = xgb.train(
            params, dtrain,
            num_boost_round=args.num_boost_round,
            evals=[(dval, "valid_0")],
            obj=train_objective,
            custom_metric=val_feval,
            maximize=True,
            early_stopping_rounds=args.early_stopping,
            verbose_eval=False,
            evals_result=evals_result,
        )
        trial.set_user_attr("best_iteration", booster.best_iteration)
        return evals_result["valid_0"]["val_f1"][booster.best_iteration]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, callbacks=[_save_progress_callback])

    print(f"\nBest val_f1: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")

    # Retrain a final model with the best params (generous round budget, same early stopping)
    final_params = _params_for_xgb(study.best_params)
    evals_result = {}
    booster = xgb.train(
        final_params, dtrain,
        num_boost_round=max(args.num_boost_round, study.best_trial.user_attrs["best_iteration"] + args.early_stopping),
        evals=[(dval, "valid_0")],
        obj=train_objective,
        custom_metric=val_feval,
        maximize=True,
        early_stopping_rounds=args.early_stopping,
        verbose_eval=25,
        evals_result=evals_result,
    )

    stats = {
        "mean": train_ds.mean.tolist(),
        "std": train_ds.std.tolist(),
        "feature_indices": feature_indices if feature_indices is not None else list(range(FEAT_DIM)),
    }
    best_ckpt = str(out_dir / "dma_xgb_tuned.xgb")
    truncated = booster[: booster.best_iteration + 1]
    DynamicWeightXGB(truncated).save(best_ckpt, stats=stats)
    print(f"Final tuned checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()

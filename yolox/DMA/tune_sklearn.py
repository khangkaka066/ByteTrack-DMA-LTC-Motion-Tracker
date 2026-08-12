"""
Bayesian hyperparameter search (Optuna, TPE) for DynamicWeightSklearn
(RandomForestRegressor / LogisticRegression).

Uses the exact same data pipeline and soft-weight training target as
train_sklearn.py — only the estimator hyperparameters are searched, so
results stay comparable with tune_gbm.py / tune_xgb.py's BO-tuned checkpoints.

Usage:
  python -m yolox.DMA.tune_sklearn \\
    --data-dir  datasets/mot17_dma \\
    --out-dir   weights/dma_sklearn_rf_mot17 \\
    --algo      rf \\
    --n-trials  50
"""

import argparse
import json
import sys
from pathlib import Path

import optuna

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from yolox.DMA.dataset import DMADataset
from yolox.DMA.model_sklearn import DynamicWeightSklearn
from yolox.DMA.train_sklearn import fit_target, evaluate
from yolox.DMA.features import FEAT_DIM
from yolox.DMA.feature_spec import resolve_feature_spec


def _suggest_rf(trial: optuna.Trial) -> dict:
    return {
        # upper bound only -- _fit_rf_early_stopping grows trees in blocks and
        # stops as soon as val F1 plateaus, so this is rarely reached in full.
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        # sklearn RF has no histogram binning like LightGBM/XGBoost, so each
        # split is an O(n log n) exact scan -- deep/unlimited trees (old range
        # went to 30) are what made RF tuning much slower than gbm/xgb/logreg.
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "n_jobs": 15,
        "random_state": 42,
    }


def _fit_rf_early_stopping(params: dict, target, train_ds, val_ds, step: int = 25, patience: int = 3):
    """Grow a RandomForest in blocks of `step` trees (via warm_start), tracking
    val F1 after each block. Stops once `patience` consecutive blocks fail to
    improve, then truncates back to the best-scoring tree count (valid for RF
    since trees are simply averaged, so dropping the trailing ones is exact,
    not an approximation).
    """
    from sklearn.ensemble import RandomForestRegressor

    params = dict(params)
    max_estimators = params.pop("n_estimators")
    estimator = RandomForestRegressor(n_estimators=step, warm_start=True, **params)

    best_score = -1.0
    best_n_estimators = step
    best_estimators_ = None
    no_improve = 0
    n = step
    while True:
        estimator.n_estimators = n
        estimator.fit(train_ds.features, target)
        score = evaluate(estimator, val_ds)
        if score > best_score + 1e-4:
            best_score, best_n_estimators = score, n
            best_estimators_ = list(estimator.estimators_)
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience or n >= max_estimators:
            break
        n = min(n + step, max_estimators)

    estimator.n_estimators = best_n_estimators
    estimator.estimators_ = best_estimators_
    return estimator, best_score


def _suggest_logreg(trial: optuna.Trial) -> dict:
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    return {
        "C": trial.suggest_float("C", 1e-4, 1e3, log=True),
        "penalty": penalty,
        "solver": solver,
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "max_iter": 2000,
        "n_jobs": 15,
    }


def _build(algo: str, params: dict):
    if algo == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**params)
    if algo == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**params)
    raise ValueError(f"Unknown --algo {algo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--val-data-dir", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--algo", required=True, choices=["rf", "logreg"])
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=None, help="seconds, optional wall-clock budget")
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

    if train_ds.motion_costs is None or train_ds.appearance_costs is None:
        raise ValueError(
            "motion_costs/appearance_costs missing from .npz data — "
            "required to build the soft-weight training target."
        )

    target = fit_target(args.algo, train_ds.motion_costs, train_ds.appearance_costs, train_ds.labels)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # create up front so progress is visible immediately

    def _save_progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        # Persist best-so-far after every trial so a crash/interrupt doesn't lose all progress
        with open(out_dir / f"best_sklearn_{args.algo}_params.json", "w") as f:
            json.dump({
                "val_f1": study.best_value,
                "params": study.best_params,
                "trials_completed": len(study.trials),
            }, f, indent=2)

    suggest_fn = _suggest_rf if args.algo == "rf" else _suggest_logreg

    def objective(trial: optuna.Trial) -> float:
        params = suggest_fn(trial)
        if args.algo == "rf":
            estimator, score = _fit_rf_early_stopping(params, target, train_ds, val_ds)
            trial.set_user_attr("n_estimators", estimator.n_estimators)
            return score
        estimator = _build(args.algo, params)
        estimator.fit(train_ds.features, target)
        return evaluate(estimator, val_ds)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, callbacks=[_save_progress_callback])

    print(f"\nBest val_f1: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")

    # Refit the final estimator with the best params
    if args.algo == "logreg":
        best_params = dict(study.best_params)
        best_params["solver"] = "liblinear" if best_params["penalty"] == "l1" else "lbfgs"
        best_params["max_iter"] = 2000
        best_params["n_jobs"] = 15
    else:
        best_params = dict(study.best_params)
        best_params.update({"n_jobs": 15, "random_state": 42})
    if args.algo == "rf":
        # replay the early-stopped tree count found during search instead of
        # re-growing all the way to the suggested n_estimators upper bound.
        best_params["n_estimators"] = study.best_trial.user_attrs["n_estimators"]
    estimator = _build(args.algo, best_params)
    estimator.fit(train_ds.features, target)
    val_f1 = evaluate(estimator, val_ds)
    print(f"Refit val_f1: {val_f1:.4f}")

    stats = {
        "mean": train_ds.mean.tolist(),
        "std": train_ds.std.tolist(),
        "feature_indices": feature_indices if feature_indices is not None else list(range(FEAT_DIM)),
    }
    best_ckpt = str(out_dir / f"dma_sklearn_{args.algo}_tuned.skl")
    DynamicWeightSklearn(estimator, args.algo).save(best_ckpt, stats=stats)
    print(f"Final tuned checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()

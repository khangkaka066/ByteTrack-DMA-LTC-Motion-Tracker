"""
scikit-learn wrapper for DMA fusion-weight prediction (RandomForest / LogisticRegression).

model_gbm.py / model_xgb.py optimise the *exact* same fused-cost BCE
objective as train.py's DynamicWeightNet, via autograd or a custom booster
objective (see gbm_objective.py). Plain sklearn estimators don't expose a
custom-objective hook generic enough for that, so this module instead uses
one of two proxy training signals (see train_sklearn.py):

  rf      RandomForestRegressor fit with MSE onto a closed-form per-sample
          target weight (soft_weight_target below) -- the exact minimiser of
          that row's fused-cost error.
  logreg  LogisticRegression fit on the binarised version of that same
          target ("should motion dominate for this pair, yes/no"); at
          inference we use predict_proba (a smooth [0, 1] value) rather than
          the hard class label, so w_motion stays continuous.

Same external contract as DynamicWeightNet / DynamicWeightGBM / DynamicWeightXGB:
  predict_numpy(x_np) -> (N, 2) [w_motion, w_reid], rows sum to 1
  save(path, stats) / load(path)
"""

import pickle

import numpy as np


def soft_weight_target(
    motion_cost: np.ndarray, appearance_cost: np.ndarray, labels: np.ndarray, eps: float = 1e-3
) -> np.ndarray:
    """Closed-form per-sample w_motion that makes the fused cost hit (1 - label) exactly."""
    target_fused = 1.0 - labels.astype(np.float64)
    mc = motion_cost.astype(np.float64)
    ac = appearance_cost.astype(np.float64)
    d = mc - ac
    # guard the near-degenerate case where motion and appearance costs agree
    # (any weight gives ~the same fused cost, so the target is ill-defined)
    safe_d = np.where(np.abs(d) < eps, np.where(d >= 0, eps, -eps), d)
    w = (target_fused - ac) / safe_d
    return np.clip(w, 0.0, 1.0).astype(np.float32)


def predict_w_motion(estimator, x_np: np.ndarray) -> np.ndarray:
    """predict_proba(class=1) for classifiers (e.g. LogisticRegression), else predict()."""
    if hasattr(estimator, "predict_proba"):
        w_motion = estimator.predict_proba(x_np)[:, 1]
    else:
        w_motion = estimator.predict(x_np)
    return np.clip(w_motion, 0.0, 1.0).astype(np.float32)


class DynamicWeightSklearn:
    """Wraps a fitted sklearn regressor/classifier predicting w_motion."""

    def __init__(self, estimator, algo: str):
        self.estimator = estimator
        self.algo = algo

    # torch-style no-ops so DMAFusion can treat this like DynamicWeightNet
    def to(self, device):
        return self

    def eval(self):
        return self

    def predict_numpy(self, x_np: np.ndarray) -> np.ndarray:
        w_motion = predict_w_motion(self.estimator, x_np)
        w_reid = 1.0 - w_motion
        return np.stack([w_motion, w_reid], axis=1)

    def save(self, path: str, stats: dict = None):
        payload = {"estimator": self.estimator, "algo": self.algo, "stats": stats}
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        stats = payload.get("stats", None)
        return cls(payload["estimator"], payload["algo"]), stats

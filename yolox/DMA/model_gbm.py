"""
LightGBM alternative to DynamicWeightNet.

Same external contract as DynamicWeightNet so it is a drop-in replacement
inside DMAFusion:
  - predict_numpy(x_np) -> (N, 2) [w_motion, w_reid], rows sum to 1
  - save(path, stats) / load(path)

The booster is trained on a single raw margin z (see gbm_objective.py);
[w_motion, w_reid] = [sigmoid(z), 1 - sigmoid(z)], which is mathematically
equivalent to softmax over 2 logits since only the logit difference matters.
"""

import pickle

import numpy as np
import lightgbm as lgb


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class DynamicWeightGBM:
    """Wraps a LightGBM Booster predicting w_motion via a single raw score."""

    def __init__(self, booster: lgb.Booster):
        self.booster = booster

    # torch-style no-ops so DMAFusion can treat this like DynamicWeightNet
    def to(self, device):
        return self

    def eval(self):
        return self

    def predict_numpy(self, x_np: np.ndarray) -> np.ndarray:
        z = self.booster.predict(x_np, raw_score=True)
        w_motion = _sigmoid(z)
        w_reid = 1.0 - w_motion
        return np.stack([w_motion, w_reid], axis=1).astype(np.float32)

    def save(self, path: str, stats: dict = None):
        payload = {"model_str": self.booster.model_to_string(), "stats": stats}
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        booster = lgb.Booster(model_str=payload["model_str"])
        stats = payload.get("stats", None)
        return cls(booster), stats

"""
XGBoost alternative DynamicWeightNet.

Same external interface as DynamicWeightNet / DynamicWeightGBM so drop-in
replacement inside DMAFusion:
  - predict_numpy(x_np) -> (N, 2) [w_motion, w_reid], rows sum 1
  - save(path, stats) / load(path)

Booster trained on single raw margin z via the exact same fused-cost BCE
custom objective as DynamicWeightGBM (see gbm_objective.py); [w_motion,
w_reid] = [sigmoid(z), 1 - sigmoid(z)].
"""

import pickle

import numpy as np
import xgboost as xgb


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class DynamicWeightXGB:
    """Wraps an xgboost.Booster predicting a single raw margin score."""

    def __init__(self, booster: xgb.Booster):
        self.booster = booster

    # torch-style no-ops so DMAFusion can treat this like DynamicWeightNet
    def to(self, device):
        return self

    def eval(self):
        return self

    def predict_numpy(self, x_np: np.ndarray) -> np.ndarray:
        dmat = xgb.DMatrix(x_np)
        z = self.booster.predict(dmat, output_margin=True)
        w_motion = _sigmoid(z)
        w_reid = 1.0 - w_motion
        return np.stack([w_motion, w_reid], axis=1).astype(np.float32)

    def save(self, path: str, stats: dict = None):
        payload = {"model_str": self.booster.save_raw(raw_format="json"), "stats": stats}
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        booster = xgb.Booster()
        booster.load_model(bytearray(payload["model_str"]))
        return cls(booster), payload.get("stats")

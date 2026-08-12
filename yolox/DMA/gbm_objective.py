"""
Custom LightGBM objective/eval mirroring train.py's BCEWeightedLoss.

The booster predicts a single raw score z (margin). w_motion = sigmoid(z),
w_reid = 1 - w_motion (equivalent to the 2-logit softmax in DynamicWeightNet,
since only the logit difference matters for a 2-class softmax).

fused = w_motion * motion_cost + w_reid * appearance_cost
      = appearance_cost + sigmoid(z) * (motion_cost - appearance_cost)

target = 1 - label   (label=1 correct match -> want fused low -> target=0)
Loss   = BCE(fused, target)

Gradient/hessian are derived by the chain rule through sigmoid(z) so that
LightGBM's Newton boosting optimizes the same fused-cost BCE objective the
MLP is trained on, rather than a generic classification loss.
"""

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def make_bce_fused_objective(motion_cost: np.ndarray, appearance_cost: np.ndarray, labels: np.ndarray):
    """
    Returns a LightGBM-compatible fobj(preds, dataset) -> (grad, hess).

    motion_cost / appearance_cost / labels must be aligned row-for-row with
    the lgb.Dataset this objective will be used to train (same order, same
    length) — LightGBM does not reorder rows across boosting rounds.
    """
    target = (1.0 - labels).astype(np.float64)
    mc = motion_cost.astype(np.float64)
    ac = appearance_cost.astype(np.float64)
    d = mc - ac

    def objective(preds: np.ndarray, train_data) -> tuple:
        z = preds.astype(np.float64)
        s = _sigmoid(z)
        f = np.clip(ac + s * d, 1e-6, 1.0 - 1e-6)
        t = target

        D = f * (1.0 - f)
        N = f - t
        dLdf = N / D
        d2Ldf2 = (D - N * (1.0 - 2.0 * f)) / (D ** 2)

        sp = s * (1.0 - s)            # ds/dz
        spp = sp * (1.0 - 2.0 * s)    # d2s/dz2

        dfdz = d * sp
        d2fdz2 = d * spp

        grad = dLdf * dfdz
        hess = d2Ldf2 * (dfdz ** 2) + dLdf * d2fdz2
        hess = np.clip(hess, 1e-3, None)  # keep Newton step well-defined
        return grad, hess

    return objective


def make_fused_eval(motion_cost: np.ndarray, appearance_cost: np.ndarray, labels: np.ndarray, name: str = "val_f1"):
    """
    Returns a LightGBM-compatible feval(preds, dataset) -> (name, value, is_higher_better)
    reporting F1 on the fused-cost < 0.5 decision, matching evaluate() in train.py.
    """
    mc = motion_cost.astype(np.float64)
    ac = appearance_cost.astype(np.float64)
    lbl = labels.astype(np.float64)

    def feval(preds: np.ndarray, train_data) -> tuple:
        z = preds.astype(np.float64)
        s = _sigmoid(z)
        fused = np.clip(ac + s * (mc - ac), 1e-6, 1.0 - 1e-6)
        pred_match = (fused < 0.5).astype(np.float64)

        tp = np.sum((pred_match == 1) & (lbl == 1))
        fp = np.sum((pred_match == 1) & (lbl == 0))
        fn = np.sum((pred_match == 0) & (lbl == 1))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return name, float(f1), True

    return feval

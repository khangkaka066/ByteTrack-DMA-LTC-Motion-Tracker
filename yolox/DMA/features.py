"""
Feature extraction for a single (track, detection) candidate pair.

Feature vector layout (FEAT_DIM = 6) — final selection, indices [1,2,3,7,10,13]
of the original 15-feature layout:
  idx  name               source
  ---  -----------------  -------
  0    motion_cost        1 - IoU(predicted_bbox, detection_bbox)
  1    mahalanobis_norm   Mahalanobis distance normalised by chi2inv95[4]
  2    cov_trace_log      log1p(trace of position covariance)
  3    cosine_dist        cosine dist between track smooth_feat & det feat
  4    bbox_area_log      log1p(w*h) / 15  (approx normalised for HD video)
  5    tracklet_len_norm  track.tracklet_len / 30  (matches --track_buffer default)

Ablation note: feature-importance and correlation analysis
(yolox/DMA/analyze_features.py, yolox/DMA/gbm_sweep.json,
sweep_gbm/*_final*.json) showed that motion_iou is perfectly anti-correlated
with motion_cost, cov_mean_log is a rescaled cov_trace_log, and
vel_magnitude / time_since_update / feat_variance / det_score / bbox_aspect /
track_age_norm / has_appearance contribute negligible val-F1 on both MOT17
and SportMOT. Dropping those 9 features (the "drop_candiate" config) changes
validation F1 by < 0.2 points on both datasets, so this 6-dimensional vector
is now the only one extracted at inference time.
"""

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from threadpoolctl import threadpool_limits

# threadpool_limits(...) as a context manager rescans every shared library
# loaded in the process (dl_iterate_phdr) on every __enter__/__exit__ — with
# torch/cv2/scipy/lightgbm all loaded that scan alone costs ~1ms, and
# extract_batch_features used to pay it every single frame just to wrap the
# tiny Mahalanobis loop below. BLAS threading only needs to be restricted
# once for the life of the process (see the comment at the call site), so
# apply it a single time here instead of re-entering a context every frame.
threadpool_limits(1, user_api="blas")

FEAT_DIM = 6

# Canonical name for each column of the (FEAT_DIM,) / (..., FEAT_DIM) vectors
# below, in order. Single source of truth for ablation tooling (sweep_gbm.py,
# analyze_features.py, feature_spec.py) so column meaning stays correct as
# the feature set evolves instead of being re-typed per script.
FEAT_NAMES = [
    "motion_cost", "mahalanobis_norm", "cov_trace_log",
    "cosine_dist", "bbox_area_log", "tracklet_len_norm",
]

# chi2inv95 for 4 degrees of freedom (used for Mahalanobis normalisation)
_CHI2_4DOF = 9.4877
_MAX_LEN = 30       # frames, for tracklet length normalisation (matches --track_buffer default)


def _iou(tlbr_a: np.ndarray, tlbr_b: np.ndarray) -> float:
    ix1 = max(tlbr_a[0], tlbr_b[0])
    iy1 = max(tlbr_a[1], tlbr_b[1])
    ix2 = min(tlbr_a[2], tlbr_b[2])
    iy2 = min(tlbr_a[3], tlbr_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a_area = max(1e-6, (tlbr_a[2] - tlbr_a[0]) * (tlbr_a[3] - tlbr_a[1]))
    b_area = max(1e-6, (tlbr_b[2] - tlbr_b[0]) * (tlbr_b[3] - tlbr_b[1]))
    union = a_area + b_area - inter
    return inter / max(union, 1e-6)


def extract_pair_features(
    track,
    detection,
    kf,
    current_frame_id: int,
) -> np.ndarray:
    """
    Extract a fixed-size feature vector for one (track, detection) pair.

    Args:
        track:            STrack (active or lost, has .mean / .covariance)
        detection:        STrack (new detection, .mean may be None)
        kf:               KalmanFilter instance (used for Mahalanobis distance)
        current_frame_id: int (unused now that track_age_norm / time_since_update
                           have been dropped; kept for interface compatibility)

    Returns:
        np.ndarray of shape (FEAT_DIM,) with dtype float32
    """
    # ── Motion ──────────────────────────────────────────────────────────────
    t_tlbr = track.tlbr           # Kalman-predicted
    d_tlbr = detection.tlbr       # raw detection

    motion_cost = 1.0 - _iou(t_tlbr, d_tlbr)

    try:
        det_xyah = detection.to_xyah().reshape(1, -1)
        maha = kf.gating_distance(
            track.mean, track.covariance, det_xyah, metric="maha"
        )[0]
        maha_norm = float(np.clip(maha / _CHI2_4DOF, 0.0, 1.0))
    except Exception:
        maha_norm = 1.0

    pos_cov = track.covariance[:4, :4]
    cov_trace_log = float(np.log1p(np.trace(pos_cov)))

    # ── Appearance ──────────────────────────────────────────────────────────
    has_app = (
        track.smooth_feat is not None
        and detection.curr_feat is not None
    )

    if has_app:
        try:
            cosine_dist = float(np.clip(
                cosine_distance(track.smooth_feat, detection.curr_feat),
                0.0, 1.0
            ))
        except Exception:
            cosine_dist = 0.5
    else:
        cosine_dist = 0.5   # neutral — no appearance signal

    # ── Detection ───────────────────────────────────────────────────────────
    tlwh = detection.tlwh
    w = max(float(tlwh[2]), 1.0)
    h = max(float(tlwh[3]), 1.0)
    bbox_area_log = float(np.log1p(w * h) / 15.0)

    # ── Track history ────────────────────────────────────────────────────────
    tracklet_len_norm = float(min(track.tracklet_len / _MAX_LEN, 1.0))

    return np.array([
        motion_cost,        # 0
        maha_norm,          # 1
        cov_trace_log,      # 2
        cosine_dist,        # 3
        bbox_area_log,      # 4
        tracklet_len_norm,  # 5
    ], dtype=np.float32)


def extract_batch_features(
    tracks: list,
    detections: list,
    kf,
    current_frame_id: int,
) -> np.ndarray:
    """
    Extract features for all (track, detection) pairs in a cost matrix.

    Vectorised equivalent of calling `extract_pair_features` in a Python
    double loop. The naive loop is O(n_tracks * n_dets) Python-level calls,
    and — critically — calls `kf.gating_distance` once per *pair*, which
    re-runs `project()` (a Cholesky factorisation of the track's covariance)
    redundantly for every detection even though it only depends on the
    track. Here each track's covariance is factorised exactly once and the
    resulting triangular solve is batched against all detections in one
    call, so Cholesky work drops from O(n_tracks * n_dets) to O(n_tracks).
    Every feature that depends on only the track or only the detection is
    computed once per track / once per detection and broadcast into the
    (n_tracks, n_dets) grid instead of being recomputed per pair.

    Returns:
        np.ndarray of shape (len(tracks), len(detections), FEAT_DIM)
    """
    n_t, n_d = len(tracks), len(detections)
    out = np.zeros((n_t, n_d, FEAT_DIM), dtype=np.float32)
    if n_t == 0 or n_d == 0:
        return out

    # ── Detection-only quantities (n_d,) ────────────────────────────────────
    # tlbr/tlwh only feed the IoU/area math below, whose result is stored in
    # a float32 output array anyway, so keeping them float32 here avoids an
    # 8-vs-4-byte upcast + copy on every frame for no precision benefit.
    det_tlbr = np.stack([d.tlbr for d in detections]).astype(np.float32, copy=False)
    det_tlwh = np.stack([d.tlwh for d in detections]).astype(np.float32, copy=False)
    # det_xyah feeds kf.gating_distance, which internally does
    # `measurements - mean` against a float64 KF mean — numpy upcasts that
    # subtraction to float64 on its own, so pre-casting here would just be a
    # redundant copy of the whole (n_d, 4) array.
    det_xyah = np.stack([d.to_xyah() for d in detections])
    det_w = np.maximum(det_tlwh[:, 2], 1.0)
    det_h = np.maximum(det_tlwh[:, 3], 1.0)
    bbox_area_log = (np.log1p(det_w * det_h) / 15.0).astype(np.float32)
    det_has_feat = np.array(
        [d.curr_feat is not None for d in detections], dtype=bool
    )

    # ── Track-only quantities (n_t,) ─────────────────────────────────────────
    t_tlbr = np.stack([t.tlbr for t in tracks]).astype(np.float32, copy=False)
    pos_cov_trace = np.array(
        [np.trace(t.covariance[:4, :4]) for t in tracks], dtype=np.float64
    )
    cov_trace_log = np.log1p(pos_cov_trace).astype(np.float32)

    tracklet_len_norm = np.minimum(
        np.array([t.tracklet_len for t in tracks], dtype=np.float64) / _MAX_LEN,
        1.0,
    ).astype(np.float32)

    t_has_feat = np.array(
        [t.smooth_feat is not None for t in tracks], dtype=bool
    )

    # ── Motion cost, fully vectorised (n_t, n_d) ────────────────────────────
    ix1 = np.maximum(t_tlbr[:, None, 0], det_tlbr[None, :, 0])
    iy1 = np.maximum(t_tlbr[:, None, 1], det_tlbr[None, :, 1])
    ix2 = np.minimum(t_tlbr[:, None, 2], det_tlbr[None, :, 2])
    iy2 = np.minimum(t_tlbr[:, None, 3], det_tlbr[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    a_area = np.maximum(
        1e-6, (t_tlbr[:, 2] - t_tlbr[:, 0]) * (t_tlbr[:, 3] - t_tlbr[:, 1])
    )
    b_area = np.maximum(
        1e-6, (det_tlbr[:, 2] - det_tlbr[:, 0]) * (det_tlbr[:, 3] - det_tlbr[:, 1])
    )
    union = a_area[:, None] + b_area[None, :] - inter
    iou = (inter / np.maximum(union, 1e-6)).astype(np.float32)
    motion_cost = 1.0 - iou

    # ── Mahalanobis: one Cholesky/solve per track, batched over detections ──
    # `kf.gating_distance` calls scipy.linalg.solve_triangular on a 4x4
    # matrix — far too small to benefit from OpenBLAS's threaded solver, but
    # OpenBLAS still pays its thread-pool sync cost on every call. Across
    # dozens of tracks/frame that overhead alone dwarfs the actual math
    # (~4ms/call vs ~0.005ms single-threaded), which is what made +DMA look
    # several times slower than ReID alone. BLAS is limited to 1 thread once
    # for the whole process at module import time (see top of file) instead
    # of re-entering threadpool_limits as a context manager every frame —
    # that scan alone cost ~1ms/call with this process's full import graph.
    maha_norm = np.ones((n_t, n_d), dtype=np.float32)
    for i, t in enumerate(tracks):
        try:
            maha = kf.gating_distance(
                t.mean, t.covariance, det_xyah, metric="maha"
            )
            maha_norm[i] = np.clip(maha / _CHI2_4DOF, 0.0, 1.0)
        except Exception:
            pass  # row already defaults to the 1.0 fallback

    # ── Appearance: pairwise cosine distance via one matmul ─────────────────
    has_app_mat = t_has_feat[:, None] & det_has_feat[None, :]
    cosine_dist = np.full((n_t, n_d), 0.5, dtype=np.float32)
    if np.any(has_app_mat):
        embed_dim = next(
            (t.smooth_feat.shape[0] for t in tracks if t.smooth_feat is not None),
            None,
        )
        if embed_dim is None:
            embed_dim = next(
                d.curr_feat.shape[0] for d in detections if d.curr_feat is not None
            )
        # Single np.stack over a list comprehension lets numpy do one
        # contiguous copy instead of n_t/n_d separate fancy-index row
        # assignments into a pre-zeroed array.
        zero_vec = np.zeros(embed_dim, dtype=np.float32)
        track_feat_mat = np.stack([
            t.smooth_feat if t.smooth_feat is not None else zero_vec
            for t in tracks
        ]).astype(np.float32, copy=False)
        det_feat_mat = np.stack([
            d.curr_feat if d.curr_feat is not None else zero_vec
            for d in detections
        ]).astype(np.float32, copy=False)

        t_unit = track_feat_mat / np.maximum(
            np.linalg.norm(track_feat_mat, axis=1, keepdims=True), 1e-12
        )
        d_unit = det_feat_mat / np.maximum(
            np.linalg.norm(det_feat_mat, axis=1, keepdims=True), 1e-12
        )
        cos_sim = t_unit @ d_unit.T
        cd = np.clip(1.0 - cos_sim, 0.0, 1.0).astype(np.float32)
        cosine_dist = np.where(has_app_mat, cd, 0.5).astype(np.float32)

    # ── Assemble (n_t, n_d, FEAT_DIM), broadcasting track-/detection-only
    # quantities across the axis they don't depend on ──────────────────────
    out[..., 0] = motion_cost
    out[..., 1] = maha_norm
    out[..., 2] = cov_trace_log[:, None]
    out[..., 3] = cosine_dist
    out[..., 4] = bbox_area_log[None, :]
    out[..., 5] = tracklet_len_norm[:, None]

    return out

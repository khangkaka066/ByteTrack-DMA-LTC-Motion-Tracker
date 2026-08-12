"""
Shared feature-selection resolver for DMA ablation tooling
(train.py / train_gbm.py / train_sklearn.py / train_xgb.py / tune_gbm.py /
tune_sklearn.py / tune_xgb.py / sweep_gbm.py).

features.py now only emits the FEAT_DIM=6 curated columns (FEAT_NAMES), so
ablating "a DMA feature" means dropping one of those 6 columns - it no
longer means indexing into the old 15-dim raw layout. This module resolves
a single --feature-indices CLI string into column indices against the
*current* FEAT_NAMES, by index or by name, so configs stay valid as the
feature set evolves instead of hardcoding indices that go stale.

Accepted --feature-indices values:
  (omitted) / "" / "all"           -> None (use all features)
  "0,2,3"                          -> keep those column indices
  "cosine_dist,tracklet_len_norm"  -> keep those named features
  "drop:cosine_dist,bbox_area_log" -> keep all features EXCEPT those named
"""

from typing import Dict, List, Optional

from yolox.DMA.features import FEAT_NAMES

_DROP_PREFIX = "drop:"


def resolve_feature_spec(
    spec: Optional[str], feat_names: List[str] = FEAT_NAMES
) -> Optional[List[int]]:
    if not spec:
        return None
    spec = spec.strip()
    if spec.lower() == "all":
        return None

    name_to_idx = {n: i for i, n in enumerate(feat_names)}

    if spec.lower().startswith(_DROP_PREFIX):
        drop_names = [s.strip() for s in spec[len(_DROP_PREFIX):].split(",") if s.strip()]
        unknown = [n for n in drop_names if n not in name_to_idx]
        if unknown:
            raise ValueError(f"Unknown feature name(s) {unknown} - choices: {feat_names}")
        drop_idx = {name_to_idx[n] for n in drop_names}
        return [i for i in range(len(feat_names)) if i not in drop_idx]

    tokens = [s.strip() for s in spec.split(",") if s.strip()]
    if all(t.lstrip("-").isdigit() for t in tokens):
        return [int(t) for t in tokens]

    unknown = [t for t in tokens if t not in name_to_idx]
    if unknown:
        raise ValueError(f"Unknown feature name(s) {unknown} - choices: {feat_names}")
    return [name_to_idx[t] for t in tokens]


def loo_configs(feat_names: List[str] = FEAT_NAMES) -> Dict[str, List[int]]:
    """Leave-one-out configs: {'drop_<name>': [all column indices except that one]}."""
    return {
        f"drop_{name}": [i for i in range(len(feat_names)) if i != j]
        for j, name in enumerate(feat_names)
    }

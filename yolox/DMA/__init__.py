from .model import DynamicWeightNet
# from .model import DynamicWeightNet6F  # uncomment if restoring FEAT_DIM=6
# from .model import DynamicWeightNet5F  # uncomment if restoring FEAT_DIM=5
from .features import extract_pair_features, FEAT_DIM
from .fuse import DMAFusion

__all__ = ["DynamicWeightNet", "DMAFusion", "extract_pair_features", "FEAT_DIM"]
# __all__ = ["DynamicWeightNet6F", "DMAFusion", "extract_pair_features", "FEAT_DIM"]  # FEAT_DIM=6
# __all__ = ["DynamicWeightNet5F", "DMAFusion", "extract_pair_features", "FEAT_DIM"]  # FEAT_DIM=5

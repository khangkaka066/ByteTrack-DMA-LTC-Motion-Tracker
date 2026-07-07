import argparse
import os
import sys


FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(FILE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.track_dma import compute_hota


def make_parser():
    parser = argparse.ArgumentParser(
        "Evaluate HOTA/CLEAR/Identity metrics from existing tracking results."
    )
    parser.add_argument(
        "--gt-root",
        type=str,
        default="datasets/sportmot/val",
        help="MOTChallenge-style GT root, e.g. datasets/sportmot/val",
    )
    parser.add_argument(
        "--results-folder",
        type=str,
        default="YOLOX_outputs/eval_reid/track_results",
        help="Folder containing tracker result .txt files",
    )
    parser.add_argument(
        "--gt-type",
        type=str,
        default="",
        help="GT suffix to evaluate, e.g. _val_half for gt_val_half.txt",
    )
    return parser


def main():
    args = make_parser().parse_args()
    gt_root = os.path.abspath(args.gt_root)
    results_folder = os.path.abspath(args.results_folder)

    if not os.path.isdir(gt_root):
        raise FileNotFoundError("GT root not found: {}".format(gt_root))
    if not os.path.isdir(results_folder):
        raise FileNotFoundError("Results folder not found: {}".format(results_folder))

    compute_hota(gt_root, results_folder, gt_type=args.gt_type)


if __name__ == "__main__":
    main()

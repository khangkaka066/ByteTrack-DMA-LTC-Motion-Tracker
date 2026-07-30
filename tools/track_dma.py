import os
import sys

FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(FILE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
from yolox.evaluators import MOTEvaluator

import argparse
import random
import warnings
import glob
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="local rank for dist training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # det args
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--slstm_ckpt", type=str, default=None, help="optional sLSTM token trajectory checkpoint")
    parser.add_argument("--slstm_vocab_size", type=int, default=256, help="sLSTM trajectory token vocabulary size")
    parser.add_argument("--slstm_context_length", type=int, default=256, help="sLSTM token context length")
    parser.add_argument("--slstm_alpha0", type=float, default=0.5, help="maximum sLSTM/Kalman blend weight")
    parser.add_argument("--slstm_beta", type=float, default=0.3, help="sLSTM blend decay for missing tracks")
    parser.add_argument("--xlstm_motion_ckpt", type=str, default=None, help="optional xLSTM motion residual checkpoint")
    parser.add_argument("--xlstm_history_len", type=int, default=16, help="xLSTM motion history length")
    parser.add_argument("--xlstm_input_dim", type=int, default=12, help="xLSTM motion history feature dimension")
    parser.add_argument("--xlstm_min_history", type=int, default=16, help="minimum history length before applying xLSTM")
    parser.add_argument("--xlstm_embedding_dim", type=int, default=128, help="xLSTM motion embedding dimension")
    parser.add_argument("--xlstm_num_blocks", type=int, default=4, help="number of xLSTM blocks")
    parser.add_argument("--xlstm_num_heads", type=int, default=4, help="number of xLSTM heads")
    parser.add_argument("--xlstm_backend", type=str, default="cuda", help="xLSTM sLSTM backend")
    parser.add_argument("--xlstm_device", type=str, default=None, help="device for xLSTM motion model")
    parser.add_argument("--xlstm_covariance_scale", type=float, default=1.0, help="scale for log_var covariance inflation")
    parser.add_argument("--xlstm_max_abs_residual", type=float, default=256.0, help="clip xLSTM residual magnitude")
    parser.add_argument("--ltc-motion-ckpt", dest="ltc_motion_ckpt", type=str, default=None, help="optional LTC/CfC motion residual checkpoint")
    parser.add_argument("--ltc_history_len", type=int, default=16, help="LTC motion history length")
    parser.add_argument("--ltc_input_dim", type=int, default=12, help="LTC motion history feature dimension")
    parser.add_argument("--ltc_min_history", type=int, default=16, help="minimum history length before applying LTC")
    parser.add_argument("--ltc_hidden_size", type=int, default=128, help="LTC hidden size")
    parser.add_argument("--ltc_num_layers", type=int, default=2, help="number of LTC/CfC layers")
    parser.add_argument("--ltc_device", type=str, default="cuda", help="device for LTC motion model")
    parser.add_argument("--ltc_covariance_scale", type=float, default=1.0, help="scale for LTC log_var covariance inflation")
    parser.add_argument("--ltc_max_abs_residual", type=float, default=256.0, help="clip LTC residual magnitude")
    # reid args
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use ReID features in ByteTrack association")
    parser.add_argument("--fast-reid", dest="fast_reid", default=False, action="store_true", help="use FastReID backend for ReID features")
    parser.add_argument("--reid-device", type=str, default="cuda", help="ReID device, e.g. cuda or cpu")
    parser.add_argument("--reid-weight", type=float, default=0.35, help="appearance cost weight when fusing IoU and ReID")
    parser.add_argument("--reid-thresh", type=float, default=0.7, help="max cosine distance allowed before ReID cost is capped")
    parser.add_argument("--reid-alpha", type=float, default=0.9, help="EMA momentum for track ReID features")
    parser.add_argument("--reid-model", type=str, default="osnet_x1_0", help="torchreid model name (deep backend)")
    parser.add_argument("--reid-model-path", type=str, default="", help="path to ReID model weights (deep backend)")
    parser.add_argument("--fast-reid-config", type=str, default="", help="FastReID config yaml")
    parser.add_argument("--fast-reid-weights", type=str, default="", help="FastReID model weights; falls back to --reid-model-path")
    parser.add_argument("--fast-reid-batch-size", type=int, default=16, help="FastReID inference batch size")
    # DMA args
    parser.add_argument("--dma-weights", type=str, default=None,
                        help="Path to trained DynamicWeightNet checkpoint (.pth). "
                             "Enables adaptive motion-appearance fusion.")
    parser.add_argument("--dma-device", type=str, default="cuda",
                        help="Device for DMA inference (cpu or cuda)")
    # video args
    parser.add_argument("--save-videos", dest="save_videos", default=False, action="store_true",
                        help="Save an annotated tracking video for each evaluated sequence")
    parser.add_argument("--video-fps", type=int, default=30, help="FPS for saved tracking videos")
    return parser


def compute_hota(gt_root, results_folder, gt_type=""):
    """Compute HOTA/DetA/AssA using the bundled TrackEval library.

    gt_root must follow MOTChallenge layout: {gt_root}/{SEQ}/gt/gt.txt
    and each sequence folder must contain a seqinfo.ini file.
    TrackEval is expected at <repo_root>/TrackEval/.
    """
    import shutil
    import tempfile

    # Add bundled TrackEval to path if not already importable
    trackeval_path = os.path.join(ROOT, "TrackEval")
    if trackeval_path not in sys.path:
        sys.path.insert(0, trackeval_path)

    try:
        import trackeval
    except ImportError:
        logger.warning("TrackEval not found at %s – skipping HOTA.", trackeval_path)
        return

    gt_root = os.path.abspath(gt_root)
    gt_filename = "gt{}.txt".format(gt_type)

    result_txts = [
        f for f in glob.glob(os.path.join(results_folder, "*.txt"))
        if not os.path.basename(f).startswith("eval")
    ]
    if not result_txts:
        logger.warning("No tracker result files found for HOTA evaluation.")
        return

    all_seq_names = [os.path.splitext(os.path.basename(f))[0] for f in result_txts]

    # Only eval sequences whose GT actually exists under gt_root
    def _seq_length(seq):
        """Return sequence length: read seqinfo.ini first, fall back to counting gt.txt rows."""
        import configparser
        ini_file = os.path.join(gt_root, seq, "seqinfo.ini")
        if os.path.isfile(ini_file):
            cfg = configparser.ConfigParser()
            cfg.read(ini_file)
            try:
                return int(cfg["Sequence"]["seqLength"])
            except (KeyError, ValueError):
                pass
        gt_file = os.path.join(gt_root, seq, "gt", gt_filename)
        if os.path.isfile(gt_file):
            with open(gt_file) as fh:
                frames = {int(line.split(",")[0]) for line in fh if line.strip()}
            return max(frames) if frames else None
        return None

    seq_info = {}
    skipped = []
    for seq, txt_file in zip(all_seq_names, result_txts):
        length = _seq_length(seq)
        if length is None:
            skipped.append(seq)
        else:
            seq_info[seq] = length

    if skipped:
        logger.warning("HOTA: skipping %d sequence(s) with no matching GT: %s", len(skipped), skipped)
    if not seq_info:
        logger.warning("HOTA: no sequences with matching GT found – skipping.")
        return

    valid_txts = {os.path.splitext(os.path.basename(f))[0]: f for f in result_txts}

    def _copy_tracker_file_for_trackeval(src_file, dst_file):
        """TrackEval treats MOT column 8 as class id; tracker outputs use -1."""
        with open(src_file, "r") as src, open(dst_file, "w") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                while len(parts) < 10:
                    parts.append("-1")
                parts[7] = "1"
                dst.write(",".join(parts[:10]) + "\n")

    def _write_trackeval_gt(seq, dst_seq_dir):
        src_gt = os.path.join(gt_root, seq, "gt", gt_filename)
        dst_gt_dir = os.path.join(dst_seq_dir, "gt")
        os.makedirs(dst_gt_dir, exist_ok=True)
        shutil.copyfile(src_gt, os.path.join(dst_gt_dir, "gt.txt"))

        src_ini = os.path.join(gt_root, seq, "seqinfo.ini")
        dst_ini = os.path.join(dst_seq_dir, "seqinfo.ini")
        if not os.path.isfile(src_ini):
            return

        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(src_ini)
        if "Sequence" not in cfg:
            cfg["Sequence"] = {}
        cfg["Sequence"]["seqLength"] = str(seq_info[seq])
        with open(dst_ini, "w") as fh:
            cfg.write(fh)

    # TrackEval tracker layout: {trackers_folder}/{tracker_name}/{sub_folder}/{SEQ}.txt
    with tempfile.TemporaryDirectory() as tmp_dir:
        trackeval_gt_root = os.path.join(tmp_dir, "gt")
        for seq in seq_info:
            _write_trackeval_gt(seq, os.path.join(trackeval_gt_root, seq))

        tracker_name = "DMA"
        tracker_data_dir = os.path.join(tmp_dir, tracker_name, "data")
        os.makedirs(tracker_data_dir, exist_ok=True)
        for seq in seq_info:
            _copy_tracker_file_for_trackeval(
                valid_txts[seq], os.path.join(tracker_data_dir, seq + ".txt")
            )

        eval_config = trackeval.Evaluator.get_default_eval_config()
        eval_config.update({
            "USE_PARALLEL": False,
            "PRINT_RESULTS": False,
            "PRINT_ONLY_COMBINED": False,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "DISPLAY_LESS_PROGRESS": True,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
        })

        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dataset_config.update({
            # SKIP_SPLIT_FOL=True → gt_fol = GT_FOLDER directly (no BENCHMARK-SPLIT subfolder)
            "GT_FOLDER": trackeval_gt_root,
            "TRACKERS_FOLDER": tmp_dir,
            "TRACKER_SUB_FOLDER": "data",
            "TRACKERS_TO_EVAL": [tracker_name],
            "SKIP_SPLIT_FOL": True,
            "SEQ_INFO": seq_info,
            "DO_PREPROC": True,
            "PRINT_CONFIG": False,
        })

        metrics_list = [
            trackeval.metrics.HOTA(),
            trackeval.metrics.CLEAR(),
            trackeval.metrics.Identity(),
        ]

        evaluator = trackeval.Evaluator(eval_config)
        dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

        logger.info("Running HOTA metrics...")
        res, _ = evaluator.evaluate(dataset_list, metrics_list)

        # Extract and print a summary table
        import numpy as np
        dataset_name = dataset_list[0].get_name()
        tracker_res = res.get(dataset_name, {}).get(tracker_name, {})
        eval_class = dataset_list[0].class_list[0] if dataset_list[0].class_list else "pedestrian"

        def _class_result(src):
            if not isinstance(src, dict):
                return {}
            return src.get(eval_class, src)

        combined = _class_result(tracker_res.get("COMBINED_SEQ", {}))

        def _mean_pct(src, metric_group, key):
            src = _class_result(src)
            v = src.get(metric_group, {}).get(key, None)
            if v is None:
                return "  N/A"
            arr = np.asarray(v, dtype=float)
            return f"{arr.mean() * 100:6.2f}" if arr.size else "  N/A"

        header = (f"{'Sequence':<30} {'HOTA':>7} {'DetA':>7} {'AssA':>7}"
                  f" {'MOTA':>7} {'MOTP':>7} {'IDF1':>7} {'IDP':>7} {'IDR':>7}")
        sep = "-" * len(header)
        lines = ["\n=== HOTA Metrics ===", sep, header, sep]

        for seq in seq_info:
            src = tracker_res.get(seq, {})
            lines.append(
                f"{seq:<30}"
                f" {_mean_pct(src, 'HOTA', 'HOTA'):>7}"
                f" {_mean_pct(src, 'HOTA', 'DetA'):>7}"
                f" {_mean_pct(src, 'HOTA', 'AssA'):>7}"
                f" {_mean_pct(src, 'CLEAR', 'MOTA'):>7}"
                f" {_mean_pct(src, 'CLEAR', 'MOTP'):>7}"
                f" {_mean_pct(src, 'Identity', 'IDF1'):>7}"
                f" {_mean_pct(src, 'Identity', 'IDP'):>7}"
                f" {_mean_pct(src, 'Identity', 'IDR'):>7}"
            )

        lines.append(sep)
        lines.append(
            f"{'OVERALL':<30}"
            f" {_mean_pct(combined, 'HOTA', 'HOTA'):>7}"
            f" {_mean_pct(combined, 'HOTA', 'DetA'):>7}"
            f" {_mean_pct(combined, 'HOTA', 'AssA'):>7}"
            f" {_mean_pct(combined, 'CLEAR', 'MOTA'):>7}"
            f" {_mean_pct(combined, 'CLEAR', 'MOTP'):>7}"
            f" {_mean_pct(combined, 'Identity', 'IDF1'):>7}"
            f" {_mean_pct(combined, 'Identity', 'IDP'):>7}"
            f" {_mean_pct(combined, 'Identity', 'IDR'):>7}"
        )
        lines.append(sep)
        print("\n".join(lines))

        def _mean_val(src, metric_group, key):
            src = _class_result(src)
            v = src.get(metric_group, {}).get(key, None)
            if v is None:
                return None
            arr = np.asarray(v, dtype=float)
            return float(arr.mean() * 100) if arr.size else None

        return {
            "HOTA": _mean_val(combined, "HOTA", "HOTA"),
            "DetA": _mean_val(combined, "HOTA", "DetA"),
            "AssA": _mean_val(combined, "HOTA", "AssA"),
            "MOTA": _mean_val(combined, "CLEAR", "MOTA"),
            "MOTP": _mean_val(combined, "CLEAR", "MOTP"),
            "IDF1": _mean_val(combined, "Identity", "IDF1"),
            "IDP": _mean_val(combined, "Identity", "IDP"),
            "IDR": _mean_val(combined, "Identity", "IDR"),
        }


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    cudnn.benchmark = True

    rank = args.local_rank
    # rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    results_folder = os.path.join(file_name, "track_results")
    os.makedirs(results_folder, exist_ok=True)
    video_folder = os.path.join(file_name, "videos")

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    #logger.info("Model Structure:\n{}".format(str(model)))

    val_loader = exp.get_eval_loader(args.batch_size, is_distributed, args.test)
    evaluator = MOTEvaluator(
        args=args,
        dataloader=val_loader,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        num_classes=exp.num_classes,
        )

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    # start evaluate
    *_, summary = evaluator.evaluate(
        model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_folder,
        video_folder=video_folder if args.save_videos else None,
        video_fps=args.video_fps,
    )
    logger.info("\n" + summary)

    # evaluate MOTA
    mm.lap.default_solver = 'lap'

    if exp.val_ann == 'val_half.json':
        gt_type = '_val_half'
    else:
        gt_type = ''
    eval_dataset = getattr(val_loader, "dataset", None)
    gt_root = None
    dataset_root = getattr(eval_dataset, "data_dir", None)
    dataset_split = getattr(eval_dataset, "name", None)
    if dataset_root and dataset_split:
        gt_root = os.path.join(dataset_root, dataset_split)

    if args.mot20:
        gt_root = gt_root or os.path.join('datasets', 'MOT20', 'train')
    else:
        gt_root = gt_root or os.path.join('datasets', 'mot', 'train')

    gtfiles = glob.glob(os.path.join(gt_root, '*/gt/gt{}.txt'.format(gt_type)))
    tsfiles = [f for f in glob.glob(os.path.join(results_folder, '*.txt')) if not os.path.basename(f).startswith('eval')]

    logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logger.info('Loading files.')

    mot_csv_sep = r'\s*,\s*'
    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1, sep=mot_csv_sep)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1, sep=mot_csv_sep)) for f in tsfiles])
    
    mh = mm.metrics.create()    
    accs, names = compare_dataframes(gt, ts)
    
    logger.info('Running metrics')
    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
               'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
               'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    # summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    # print(mm.io.render_summary(
    #   summary, formatters=mh.formatters, 
    #   namemap=mm.io.motchallenge_metric_names))
    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])
    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                       'partially_tracked', 'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']
    print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

    # HOTA metrics (requires trackeval; skipped gracefully if not installed)
    hota_metrics = compute_hota(gt_root, results_folder, gt_type=gt_type)

    import json
    overall_mot = summary.loc['OVERALL'].to_dict() if 'OVERALL' in summary.index else {}
    with open(os.path.join(file_name, 'summary.json'), 'w') as f:
        json.dump({
            "experiment_name": args.experiment_name,
            "hota": hota_metrics,
            "motmetrics": overall_mot,
        }, f, indent=2, default=str)

    logger.info('Completed')


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    if args.fast_reid:
        if not args.fast_reid_config:
            parser.error("--fast-reid-config is required with --fast-reid")
        if not args.fast_reid_weights and not args.reid_model_path:
            parser.error("--fast-reid-weights or --reid-model-path is required with --fast-reid")
    args.reid_backend = "fast" if args.fast_reid else "deep"
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        if args.slstm_ckpt:
            args.experiment_name = "eval_slstm"
        elif args.xlstm_motion_ckpt:
            args.experiment_name = "eval_xlstm"
        elif args.ltc_motion_ckpt:
            args.experiment_name = "eval_ltc"
        elif args.with_reid:
            args.experiment_name = "eval_reid"
        else:
            args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu),
    )

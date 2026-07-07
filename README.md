# ByteTrack — Extended Fork

This repository is a research fork of [ByteTrack](https://github.com/ifzhang/ByteTrack)
(*ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022).
The core BYTE association and the YOLOX detection pipeline are kept, but the
codebase has diverged from the original paper with several learned components
plugged into (or around) the Kalman motion model:

- **xLSTM motion residual** — an NX-AI xLSTM model that refines the Kalman
  prediction with a learned residual and uncertainty
  (`yolox/tracker/xlstm_motion.py`, see [docs/xlstm_motion_kalman.md](docs/xlstm_motion_kalman.md)).
- **LTC/CfC motion residual** — a continuous-time Liquid Time-Constant / CfC
  residual predictor with the same role (`yolox/tracker/ltc_motion.py`,
  see [LTC_MOTION_KAGGLE.md](LTC_MOTION_KAGGLE.md)).
- **sLSTM token trajectory predictor** — a token-based trajectory model blended
  with the Kalman prediction (`yolox/xlstm/`).
- **ReID appearance features** — optional appearance cost in the BYTE
  association, with a torchreid (OSNet) backend or a
  [FastReID](fast-reid/) backend (`yolox/tracker/reid.py`, `yolox/tracker_reid/`).
- **DMA (Dynamic Motion-Appearance fusion)** — a small `DynamicWeightNet` that
  adaptively weighs motion vs. appearance cost per track/detection pair
  (`yolox/DMA/`, `tools/track_dma.py`).

The math behind the baseline tracker and the LTC extension is documented in
[`bytetrack_math.pdf`](bytetrack_math.pdf) (source: `bytetrack_math.tex`).

> Because of these changes, results are **not** expected to reproduce the
> numbers of the original ByteTrack paper.

## Repository structure

```text
.
├── assets/                     # demo GIFs / images
├── datasets/                   # place MOT17 / MOT20 / CrowdHuman / DanceTrack here
├── deploy/                     # TensorRT (C++/Python), ONNX, ncnn deployment
├── docs/
│   └── xlstm_motion_kalman.md  # xLSTM motion: data format, loss, ckpt, inference
├── exps/
│   ├── default/                # standard YOLOX experiment configs
│   └── example/mot/            # MOT experiment configs (yolox_x_mix_det.py, ...)
├── fast-reid/                  # vendored FastReID (optional ReID backend)
├── tools/                      # all entry points (see below)
├── tutorials/                  # applying BYTE to other trackers
├── videos/                     # demo input videos
├── xlstm/                      # vendored NX-AI xLSTM library
└── yolox/
    ├── core/ | data/ | evaluators/ | exp/ | layers/ | models/ | utils/   # YOLOX detector
    ├── tracker/                # ByteTrack + kalman_filter, xlstm_motion, ltc_motion, reid
    ├── tracker_reid/           # ByteTrack variant with ReID-fused association
    ├── xlstm/                  # sLSTM token trajectory predictor + byte_tracker_slstm
    ├── DMA/                    # DynamicWeightNet: features, dataset gen, training, fusion
    ├── deepsort_tracker/ | sort_tracker/ | motdt_tracker/   # baseline trackers
    └── tracking_utils/
```

### Entry points (`tools/`)

| Script | Purpose |
| --- | --- |
| `train.py` | Train the YOLOX detector |
| `track.py` | Evaluate ByteTrack (supports sLSTM / xLSTM / LTC / ReID / DMA flags) |
| `track_reid.py` | Evaluate the ReID-fused tracker (`yolox/tracker_reid`) |
| `track_dma.py` | Evaluate ByteTrack with DMA adaptive fusion |
| `track_sort.py`, `track_deepsort.py`, `track_motdt.py` | Baseline trackers with the same detector |
| `demo_track.py` | Run tracking on a video / images / webcam |
| `train_xlstm_motion.py` | Train the xLSTM motion residual model |
| `train_ltc_motion.py` | Train the LTC/CfC motion residual model |
| `interpolation.py` | Offline track interpolation post-processing |
| `mota.py`, `txt2video.py`, `convert_video.py` | Evaluation / visualization helpers |
| `convert_*_to_coco.py`, `mix_data_*.py` | Dataset conversion and mixing |
| `download_bytetrack_weights.py` | Download pretrained ByteTrack YOLOX weights |
| `export_onnx.py`, `trt.py` | Model export / TensorRT |

## Installation

```bash
git clone <this repo>
cd bytetrack
pip install -r requirements.txt
python setup.py develop
pip install cython pycocotools cython_bbox
```

Optional, depending on which extensions you use:

- **xLSTM / sLSTM**: the vendored `xlstm/` package plus a CUDA-capable PyTorch
  (`--xlstm_backend cuda`; use `vanilla` for CPU).
- **torchreid backend**: `pip install torchreid` (OSNet weights via `--reid-model-path`).
- **FastReID backend**: install from the vendored `fast-reid/` directory.

Kaggle- and Vast.ai-specific setup lives in [KAGGLE.md](KAGGLE.md),
`requirements-kaggle.txt`, [VAST_GPU_DANCETRACK_TRAINING.txt](VAST_GPU_DANCETRACK_TRAINING.txt)
and the notebook `finetune_yolox_m_dancetrack_container.ipynb`.

## Data preparation

Download [MOT17](https://motchallenge.net/), [MOT20](https://motchallenge.net/),
[CrowdHuman](https://www.crowdhuman.org/), Cityperson, ETHZ and put them under
`datasets/`, then convert to COCO format and create mixed training sets:

```bash
python tools/convert_mot17_to_coco.py
python tools/convert_mot20_to_coco.py
python tools/convert_crowdhuman_to_coco.py
python tools/convert_cityperson_to_coco.py
python tools/convert_ethz_to_coco.py

python tools/mix_data_ablation.py
python tools/mix_data_test_mot17.py
python tools/mix_data_test_mot20.py
```

Pretrained ByteTrack detector weights can be fetched with:

```bash
python tools/download_bytetrack_weights.py
```

## Detector training

```bash
# example: train yolox-x on the MOT17 mix
python tools/train.py -f exps/example/mot/yolox_x_mix_det.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
```

Experiment configs for all model sizes (nano → x) live in `exps/example/mot/`.

## Tracking

### Baseline ByteTrack

```bash
python tools/track.py -f exps/example/mot/yolox_x_mix_det.py \
    -c pretrained/bytetrack_x_mot17.pth.tar -b 1 -d 1 --fp16 --fuse
python tools/interpolation.py   # optional offline interpolation
```

For MOT20 add `--mot20 --match_thresh 0.7` and use `yolox_x_mix_mot20_ch.py`.

### xLSTM motion residual

Train, then pass the checkpoint at tracking time:

```bash
python tools/train_xlstm_motion.py --data-root datasets/mot_frcnn/train --output outputs/xlstm_motion.pth
python tools/track.py ... --xlstm_motion_ckpt outputs/xlstm_motion.pth
```

Full details (feature layout, loss, checkpoint format, all
`--xlstm_*` flags) are in [docs/xlstm_motion_kalman.md](docs/xlstm_motion_kalman.md).

### LTC/CfC motion residual

```bash
python tools/train_ltc_motion.py --data-root datasets/mot/train --output outputs/ltc_motion.pth
python tools/track.py ... --ltc_motion_ckpt outputs/ltc_motion.pth
```

See [LTC_MOTION_KAGGLE.md](LTC_MOTION_KAGGLE.md) for the Kaggle walkthrough and
the `--ltc_*` tuning flags.

### sLSTM token trajectory blending

```bash
python tools/track.py ... --slstm_ckpt <ckpt> --slstm_alpha0 0.5 --slstm_beta 0.3
```

### ReID-fused association

Either through the flags on `tools/track.py`, or the dedicated tracker:

```bash
# torchreid / OSNet backend
python tools/track_reid.py ... --with-reid --reid-weight 0.35 --reid-thresh 0.7

# FastReID backend
python tools/track_reid.py ... --with-reid --fast-reid \
    --fast-reid-config <cfg.yaml> --fast-reid-weights <model.pth>
```

### DMA — adaptive motion/appearance fusion

Generate pairwise training data from MOT GT, train `DynamicWeightNet`, then track:

```bash
python -m yolox.DMA.generate_data --seq-dirs datasets/mot/train/MOT17-02-FRCNN ... --out-dir data/dma_train
python -m yolox.DMA.train --data-dir data/dma_train --out-dir outputs/dma
python tools/track_dma.py ... --dma-weights outputs/dma/best.pth --with-reid
```

## Demo

```bash
python tools/demo_track.py video -f exps/example/mot/yolox_x_mix_det.py \
    -c pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result
```

The same motion/ReID flags as `tools/track.py` are available.

## Deployment

ONNX, TensorRT (C++ and Python) and ncnn exports are under `deploy/`;
`tools/export_onnx.py` and `tools/trt.py` are the entry points. See the
READMEs inside `deploy/` for details. `tutorials/` shows how to apply the BYTE
association to other trackers.

## Citation & acknowledgement

This fork builds on:

- [ByteTrack](https://github.com/ifzhang/ByteTrack) (Zhang et al., ECCV 2022)

```bibtex
@article{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) for the detector
- [NX-AI xLSTM](https://github.com/NX-AI/xlstm) for the xLSTM library (vendored in `xlstm/`)
- [FastReID](https://github.com/JDAI-CV/fast-reid) (vendored in `fast-reid/`) and
  [torchreid](https://github.com/KaiyangZhou/deep-person-reid) for ReID backends
- [FairMOT](https://github.com/ifzhang/FairMOT), [TransTrack](https://github.com/PeizeSun/TransTrack)
  for parts of the tracking/evaluation code

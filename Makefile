# Quick entry points around tools/track_dma.py so common evaluation runs
# don't need long CLI invocations. Run `make help` for the list of targets.
#
# Every target accepts:
#   EXPN=<name>   override the experiment name (default set per target)
#   ARGS="..."    extra args appended verbatim to track_dma.py
# e.g. `make track-mot17-dma EXPN=my_run ARGS="--track_thresh 0.5"`

PY := python3

# ---- MOT17 --------------------------------------------------------------
EXP_MOT17      := exps/example/mot/yolox_x_mot17_half.py
CKPT_MOT17     := pretrained/bytetrack_x_mot17.pth.tar
FR_CFG_MOT17   := fast-reid/configs/MOT17/sbs_S50.yml
FR_W_MOT17     := reid_weights/mot17_sbs_S50.pth
TR_MODEL_MOT17 := osnet_x0_25
TR_W_MOT17     := reid_weights/osnet_good/osnet_x0_25_msmt17_combineall_good_inference.pth
DMA_MOT17      := dma_weights/gbm_gpu/dma_gbm_tuned.gbm
LTC_MOT17      := ltc_weights/ltc_motion_mot17_duplicate.pth

# ---- SportsMOT ------------------------------------------------------------
EXP_SPORTSMOT    := exps/example/sportmot/yolox_x_sportsmot.py
CKPT_SPORTSMOT   := pretrained/yolox_x_sports_mix.pth.tar
FR_CFG_SPORTSMOT := fast-reid/configs/SportMOT/sbs_S50.yml
FR_W_SPORTSMOT   := reid_weights/sportmot_sbs_S50.pth
TR_MODEL_SPORTSMOT := osnet_x1_0
TR_W_SPORTSMOT   := reid_weights/osnet/osnet_x1_0_msmt17_combineall.pth
DMA_SPORTSMOT    := dma_weights/gbm_sportmot/val/dma_gbm_tuned.gbm
LTC_SPORTSMOT    := ltc_weights/ltc_motion_sportsmot.pth

# ---- Tune GBM ------------------------------------------------------------
DATA_DIR := datasets/mot17_dma_train
OUT_DIR := dma_weights/gbm_gpu
DEVICE := cuda

COMMON_ARGS := -b 1 -d 1 --fuse --fp16

.PHONY: help \
	track-mot17 track-mot17-reid track-mot17-reid-torch track-mot17-dma track-mot17-dma-torch \
	track-mot17-ltc track-mot17-reid-ltc track-mot17-reid-ltc-torch track-mot17-full track-mot17-full-torch \
	track-sportsmot track-sportsmot-reid track-sportsmot-reid-torch track-sportsmot-dma track-sportsmot-dma-torch \
	track-sportsmot-ltc track-sportsmot-reid-ltc track-sportsmot-reid-ltc-torch track-sportsmot-full track-sportsmot-full-torch \
	clean

help:
	@echo "Available targets (all wrap tools/track_dma.py):"
	@echo ""
	@echo "  MOT17:       track-mot17 track-mot17-reid[-torch] track-mot17-dma[-torch] track-mot17-ltc track-mot17-reid-ltc[-torch] track-mot17-full[-torch]"
	@echo "  SportsMOT:   track-sportsmot track-sportsmot-reid[-torch] track-sportsmot-dma[-torch] track-sportsmot-ltc track-sportsmot-reid-ltc[-torch] track-sportsmot-full[-torch]"
	@echo ""
	@echo "  '-torch' variants use the torchreid (OSNet) backend instead of FastReID;"
	@echo "  the default (no suffix) targets use FastReID."
	@echo ""
	@echo "  clean EXPN=<name>   remove YOLOX_outputs/<name>"
	@echo ""
	@echo "Override with EXPN=<name> and pass extra flags with ARGS=\"...\"."

# ---------------------------------------------------------------------------
# MOT17
# ---------------------------------------------------------------------------
track-mot17: EXPN ?= eval_mot17_baseline
track-mot17:
	$(PY) tools/track_dma.py -f $(EXP_MOT17) -c $(CKPT_MOT17) -expn $(EXPN) $(COMMON_ARGS) $(ARGS)

track-mot17-reid: EXPN ?= eval_mot17_reid
track-mot17-reid:
	$(PY) tools/track_dma.py -f $(EXP_MOT17) -c $(CKPT_MOT17) -expn $(EXPN) $(COMMON_ARGS) \
		--with-reid --fast-reid --fast-reid-config $(FR_CFG_MOT17) --fast-reid-weights $(FR_W_MOT17) $(ARGS)

track-mot17-reid-torch: EXPN ?= eval_mot17_reid_torch
track-mot17-reid-torch:
	$(PY) tools/track_dma.py -f $(EXP_MOT17) -c $(CKPT_MOT17) -expn $(EXPN) $(COMMON_ARGS) \
		--with-reid --reid-model $(TR_MODEL_MOT17) --reid-model-path $(TR_W_MOT17) $(ARGS)

track-mot17-dma: EXPN ?= eval_mot17_dma
track-mot17-dma:
	$(PY) tools/track_dma.py -f $(EXP_MOT17) -c $(CKPT_MOT17) -expn $(EXPN) $(COMMON_ARGS) \
		--with-reid --fast-reid --fast-reid-config $(FR_CFG_MOT17) --fast-reid-weights $(FR_W_MOT17) \
		--ml gbm --ml-weights $(DMA_MOT17) $(ARGS)

track-mot17-dma-torch: EXPN ?= eval_mot17_dma_torch
track-mot17-dma-torch:
	$(PY) tools/track_dma.py -f $(EXP_MOT17) -c $(CKPT_MOT17) -expn $(EXPN) $(COMMON_ARGS) \
		--with-reid --reid-model $(TR_MODEL_MOT17) --reid-model-path $(TR_W_MOT17) \
		--ml gbm --ml-weights $(DMA_MOT17) $(ARGS)

track-mot17-ltc: EXPN ?= eval_mot17_ltc
track-mot17-ltc:
	$(PY) tools/track_dma.py -f $(EXP_MOT17) -c $(CKPT_MOT17) -expn $(EXPN) $(COMMON_ARGS) \
		--ltc-motion-ckpt $(LTC_MOT17) $(ARGS)

track-mot17-reid-ltc: EXPN ?= eval_mot17_reid_ltc
track-mot17-reid-ltc:
	$(PY) tools/track_dma.py -f $(EXP_MOT17) -c $(CKPT_MOT17) -expn $(EXPN) $(COMMON_ARGS) \
		--with-reid --fast-reid --fast-reid-config $(FR_CFG_MOT17) --fast-reid-weights $(FR_W_MOT17) \
		--ltc-motion-ckpt $(LTC_MOT17) $(ARGS)

track-mot17-reid-ltc-torch: EXPN ?= eval_mot17_reid_ltc_torch
track-mot17-reid-ltc-torch:
	$(PY) tools/track_dma.py -f $(EXP_MOT17) -c $(CKPT_MOT17) -expn $(EXPN) $(COMMON_ARGS) \
		--with-reid --reid-model $(TR_MODEL_MOT17) --reid-model-path $(TR_W_MOT17) \
		--ltc-motion-ckpt $(LTC_MOT17) $(ARGS)

track-mot17-full: EXPN ?= eval_mot17_full
track-mot17-full:
	$(PY) tools/track_dma.py -f $(EXP_MOT17) -c $(CKPT_MOT17) -expn $(EXPN) $(COMMON_ARGS) \
		--with-reid --fast-reid --fast-reid-config $(FR_CFG_MOT17) --fast-reid-weights $(FR_W_MOT17) \
		--ml gbm --ml-weights $(DMA_MOT17) --ltc-motion-ckpt $(LTC_MOT17) $(ARGS)

track-mot17-full-torch: EXPN ?= eval_mot17_full_torch
track-mot17-full-torch:
	$(PY) tools/track_dma.py -f $(EXP_MOT17) -c $(CKPT_MOT17) -expn $(EXPN) $(COMMON_ARGS) \
		--with-reid --reid-model $(TR_MODEL_MOT17) --reid-model-path $(TR_W_MOT17) \
		--ml gbm --ml-weights $(DMA_MOT17) --ltc-motion-ckpt $(LTC_MOT17) $(ARGS)

# ---------------------------------------------------------------------------
# SportsMOT
# ---------------------------------------------------------------------------
track-sportsmot: EXPN ?= eval_sportsmot_baseline
track-sportsmot:
	$(PY) tools/track_dma.py -f $(EXP_SPORTSMOT) -c $(CKPT_SPORTSMOT) -expn $(EXPN) $(COMMON_ARGS) $(ARGS)

track-sportsmot-reid: EXPN ?= eval_sportsmot_reid
track-sportsmot-reid:
	$(PY) tools/track_dma.py -f $(EXP_SPORTSMOT) -c $(CKPT_SPORTSMOT) -expn $(EXPN) $(COMMON_ARGS) \
		--with-reid --fast-reid --fast-reid-config $(FR_CFG_SPORTSMOT) --fast-reid-weights $(FR_W_SPORTSMOT) $(ARGS)

track-sportsmot-reid-torch: EXPN ?= eval_sportsmot_reid_torch
track-sportsmot-reid-torch:
	$(PY) tools/track_dma.py -f $(EXP_SPORTSMOT) -c $(CKPT_SPORTSMOT) -expn $(EXPN) $(COMMON_ARGS) \
		--with-reid --reid-model $(TR_MODEL_SPORTSMOT) --reid-model-path $(TR_W_SPORTSMOT) $(ARGS)

track-sportsmot-dma: EXPN ?= eval_sportsmot_dma
track-sportsmot-dma:
	$(PY) tools/track_dma.py -f $(EXP_SPORTSMOT) -c $(CKPT_SPORTSMOT) -expn $(EXPN) $(COMMON_ARGS) \
		--with-reid --fast-reid --fast-reid-config $(FR_CFG_SPORTSMOT) --fast-reid-weights $(FR_W_SPORTSMOT) \
		--ml gbm --ml-weights $(DMA_SPORTSMOT) $(ARGS)

track-sportsmot-dma-torch: EXPN ?= eval_sportsmot_dma_torch
track-sportsmot-dma-torch:
	$(PY) tools/track_dma.py -f $(EXP_SPORTSMOT) -c $(CKPT_SPORTSMOT) -expn $(EXPN) $(COMMON_ARGS) \
		--with-reid --reid-model $(TR_MODEL_SPORTSMOT) --reid-model-path $(TR_W_SPORTSMOT) \
		--ml gbm --ml-weights $(DMA_SPORTSMOT) $(ARGS)

track-sportsmot-ltc: EXPN ?= eval_sportsmot_ltc
track-sportsmot-ltc:
	$(PY) tools/track_dma.py -f $(EXP_SPORTSMOT) -c $(CKPT_SPORTSMOT) -expn $(EXPN) $(COMMON_ARGS) \
		--ltc-motion-ckpt $(LTC_SPORTSMOT) $(ARGS)

track-sportsmot-reid-ltc: EXPN ?= eval_sportsmot_reid_ltc
track-sportsmot-reid-ltc:
	$(PY) tools/track_dma.py -f $(EXP_SPORTSMOT) -c $(CKPT_SPORTSMOT) -expn $(EXPN) $(COMMON_ARGS) \
		--with-reid --fast-reid --fast-reid-config $(FR_CFG_SPORTSMOT) --fast-reid-weights $(FR_W_SPORTSMOT) \
		--ltc-motion-ckpt $(LTC_SPORTSMOT) $(ARGS)

track-sportsmot-reid-ltc-torch: EXPN ?= eval_sportsmot_reid_ltc_torch
track-sportsmot-reid-ltc-torch:
	$(PY) tools/track_dma.py -f $(EXP_SPORTSMOT) -c $(CKPT_SPORTSMOT) -expn $(EXPN) $(COMMON_ARGS) \
		--with-reid --reid-model $(TR_MODEL_SPORTSMOT) --reid-model-path $(TR_W_SPORTSMOT) \
		--ltc-motion-ckpt $(LTC_SPORTSMOT) $(ARGS)

track-sportsmot-full: EXPN ?= eval_sportsmot_full
track-sportsmot-full:
	$(PY) tools/track_dma.py -f $(EXP_SPORTSMOT) -c $(CKPT_SPORTSMOT) -expn $(EXPN) $(COMMON_ARGS) \
		--with-reid --fast-reid --fast-reid-config $(FR_CFG_SPORTSMOT) --fast-reid-weights $(FR_W_SPORTSMOT) \
		--ml gbm --ml-weights $(DMA_SPORTSMOT) --ltc-motion-ckpt $(LTC_SPORTSMOT) $(ARGS)

track-sportsmot-full-torch: EXPN ?= eval_sportsmot_full_torch
track-sportsmot-full-torch:
	$(PY) tools/track_dma.py -f $(EXP_SPORTSMOT) -c $(CKPT_SPORTSMOT) -expn $(EXPN) $(COMMON_ARGS) \
		--with-reid --reid-model $(TR_MODEL_SPORTSMOT) --reid-model-path $(TR_W_SPORTSMOT) \
		--ml gbm --ml-weights $(DMA_SPORTSMOT) --ltc-motion-ckpt $(LTC_SPORTSMOT) $(ARGS)


# ---------------------------------------------------------------------------
# Tune GBM
# ---------------------------------------------------------------------------

tune:
	$(PY) yolox/DMA/tune_gbm.py --data-dir $(DATA_DIR) --out-dir $(OUT_DIR) --device $(DEVICE)

# ---------------------------------------------------------------------------
clean:
	@test -n "$(EXPN)" || (echo "usage: make clean EXPN=<name>"; exit 1)
	rm -rf "YOLOX_outputs/$(EXPN)"

"""Plot the detector benchmark produced by tools/benchmark_detectors.py.

Writes <out_dir>/detector_benchmark.png with:
  row 1  accuracy vs latency (HOTA, IDF1) with the Pareto front
  row 2  per-metric bars (HOTA, IDF1, MOTA, detection AP50:95, ID switches)
  row 3  end-to-end FPS (detector + tracker)

Usage:
    python tools/plot_detector_benchmark.py YOLOX_outputs/detbench_museum/results.json
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e6e5e1"
HIGHLIGHT = "#2a78d6"
DIM = "#a9c6ec"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _pareto(rows, key):
    """Models not beaten by another that is both faster and more accurate."""
    front = []
    for r in sorted(rows, key=lambda r: r["total_ms"]):
        if not front or r[key] > front[-1][key]:
            front.append(r)
    return front


def _scatter(ax, rows, key, label):
    _style(ax)
    front = _pareto(rows, key)
    names = {r["model"] for r in front}
    ax.plot([r["total_ms"] for r in front], [r[key] for r in front],
            color=MUTED, linewidth=1.2, linestyle="--", zorder=1, label="Pareto front")
    for r in rows:
        on = r["model"] in names
        ax.scatter(r["total_ms"], r[key], s=70, color=HIGHLIGHT if on else DIM,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(r["model"], (r["total_ms"], r[key]), xytext=(6, 5), textcoords="offset points",
                    fontsize=9, color=INK if on else INK_2)
    ax.set_xlabel("Latency per frame (ms, detector + tracker) - lower is better", color=INK_2, fontsize=9)
    ax.set_ylabel(f"{label} (%)", color=INK_2, fontsize=9)
    ax.set_title(f"{label} vs latency", loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.margins(x=0.18, y=0.15)
    if len(front) > 1:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    else:
        ax.text(0.99, 0.02, f"{front[0]['model']} is both fastest and most accurate",
                transform=ax.transAxes, ha="right", fontsize=8, color=INK_2)


def _bars(ax, rows, key, label, order, lower_better=False, fmt="{:.1f}"):
    _style(ax)
    ax.grid(axis="y", visible=False)
    vals = {r["model"]: r[key] or 0 for r in rows}
    best = (min if lower_better else max)(vals.values())
    y = range(len(order))
    colors = [HIGHLIGHT if vals[m] == best else DIM for m in order]
    ax.barh(list(y), [vals[m] for m in order], color=colors, height=0.62)
    ax.set_yticks(list(y))
    ax.set_yticklabels(order, color=INK_2)
    ax.invert_yaxis()
    xmax = max(vals.values()) or 1
    for i, m in enumerate(order):
        ax.text(vals[m] + xmax * 0.02, i, fmt.format(vals[m]), va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, xmax * 1.22)
    arrow = "lower is better" if lower_better else "higher is better"
    ax.set_title(f"{label}\n", loc="left", color=INK, fontsize=10.5, fontweight="bold")
    ax.text(0, 1.02, arrow, transform=ax.transAxes, fontsize=8, color=MUTED)


def _fps(ax, rows, order):
    _style(ax)
    ax.grid(axis="y", visible=False)
    by = {r["model"]: r for r in rows}
    y = list(range(len(order)))
    fps = [by[m]["fps"] for m in order]
    best = max(fps)
    ax.barh(y, fps, color=[HIGHLIGHT if v == best else DIM for v in fps], height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(order, color=INK_2)
    ax.invert_yaxis()
    for i, m in enumerate(order):
        r = by[m]
        ax.text(r["fps"] + best * 0.01, i, f"{r['fps']:.1f} FPS  ({r['total_ms']:.1f} ms)",
                va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, best * 1.2)
    ax.set_xlabel("End-to-end FPS (detector + tracker, batch 1)", color=INK_2, fontsize=9)
    ax.set_title("Throughput (FPS)\n", loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.text(0, 1.02, "higher is better", transform=ax.transAxes, fontsize=8, color=MUTED)


def plot(results_json, out_dir=None):
    with open(results_json) as fh:
        rows = [r for r in json.load(fh) if r.get("HOTA") is not None and r.get("total_ms")]
    if not rows:
        raise SystemExit("nothing to plot")
    out_dir = out_dir or os.path.dirname(results_json)
    order = [r["model"] for r in sorted(rows, key=lambda r: -r["HOTA"])]
    r0 = rows[0]
    subtitle = (f"MuseumVisitors held-out frames - ByteTrack only (no ReID / DMA / motion models) - "
                f"input {r0.get('test_size')} - {'FP16' if r0.get('fp16') else 'FP32'} - batch 1")

    fig = plt.figure(figsize=(16, 13), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 5, height_ratios=[1.25, 1, 0.8], hspace=0.55, wspace=0.55)
    _scatter(fig.add_subplot(gs[0, :3]), rows, "HOTA", "HOTA")
    _scatter(fig.add_subplot(gs[0, 3:]), rows, "IDF1", "IDF1")
    metrics = [("HOTA", "HOTA", False, "{:.1f}"), ("IDF1", "IDF1", False, "{:.1f}"),
               ("MOTA", "MOTA", False, "{:.1f}"), ("AP50_95", "Det AP50:95", False, "{:.1f}"),
               ("IDSW", "ID switches", True, "{:.0f}")]
    for i, (key, label, low, fmt) in enumerate(metrics):
        _bars(fig.add_subplot(gs[1, i]), rows, key, label, order, low, fmt)
    _fps(fig.add_subplot(gs[2, :]), rows, order)

    fig.suptitle("Detector benchmark for tracking", x=0.06, y=0.975, ha="left",
                 fontsize=16, fontweight="bold", color=INK)
    fig.text(0.06, 0.948, subtitle, fontsize=9.5, color=INK_2)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.05)
    out = os.path.join(out_dir, "detector_benchmark.png")
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(f"chart: {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("results_json")
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()
    plot(a.results_json, a.out_dir)

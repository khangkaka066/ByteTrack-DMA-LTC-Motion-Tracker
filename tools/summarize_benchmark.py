"""Aggregate summary.json files produced by tools/track_dma.py runs into one
comparison table across tracking methods.

Usage:
    python3 tools/summarize_benchmark.py YOLOX_outputs/method_a YOLOX_outputs/method_b ...
"""
import argparse
import json
import os


COLUMNS = [
    ("HOTA", "hota", "HOTA"),
    ("DetA", "hota", "DetA"),
    ("AssA", "hota", "AssA"),
    ("MOTA", "motmetrics", "mota"),
    ("IDF1", "motmetrics", "idf1"),
    ("IDSW", "motmetrics", "num_switches"),
]


def load_summary(exp_dir):
    path = os.path.join(exp_dir, "summary.json")
    if not os.path.isfile(path):
        print(f"[skip] no summary.json in {exp_dir}")
        return None
    with open(path) as f:
        return json.load(f)


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dirs", nargs="+", help="experiment output dirs, e.g. YOLOX_outputs/<name>")
    args = parser.parse_args()

    rows = []
    for exp_dir in args.exp_dirs:
        data = load_summary(exp_dir)
        if data is None:
            continue
        name = data.get("experiment_name", os.path.basename(exp_dir))
        row = [name]
        for _, group, key in COLUMNS:
            row.append(fmt(data.get(group, {}).get(key)))
        rows.append(row)

    if not rows:
        print("No summaries found.")
        return

    header = ["Method"] + [c[0] for c in COLUMNS]
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]

    def render(row):
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)
    print(render(header))
    print(sep)
    for row in rows:
        print(render(row))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stack tracking result videos from multiple method output folders side by side
for visual comparison.

Each method folder under YOLOX_outputs/<method>/videos/<name>.mp4 must contain
videos with the same file names. For every video name present in ALL method
folders, this script builds a grid (one tile per method, labeled with the
method name) using ffmpeg's xstack filter and writes the result to
YOLOX_outputs/video_cmp/<name>.mp4.
"""
import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_METHODS = [
    "baseline_bytetrack",
    "with_reid_0_856",
    "with_dma",
    "with_ltc",
    "ltc_reid_0_856",
    "ltc_reid_dma",
]


def find_common_videos(root: Path, methods: list[str]) -> list[str]:
    sets = []
    for m in methods:
        vdir = root / m / "videos"
        if not vdir.is_dir():
            raise FileNotFoundError(f"Missing videos dir: {vdir}")
        sets.append({p.name for p in vdir.glob("*.mp4")})
    common = set.intersection(*sets)
    return sorted(common)


def grid_dims(n: int) -> tuple[int, int]:
    """Return (cols, rows) as close to square as possible, cols >= rows."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows


def build_filter_complex(n: int, methods: list[str], tile_w: int, tile_h: int) -> tuple[str, list[str]]:
    cols, rows = grid_dims(n)
    parts = []
    labeled = []
    for i, method in enumerate(methods):
        label = method.replace("'", "")
        parts.append(
            f"[{i}:v]scale={tile_w}:{tile_h},"
            f"drawtext=text='{label}':x=10:y=10:fontsize=28:fontcolor=white:"
            f"box=1:boxcolor=black@0.5:boxborderw=6[v{i}]"
        )
        labeled.append(f"[v{i}]")

    layout_positions = []
    for i in range(n):
        col = i % cols
        row = i // cols
        x = "0" if col == 0 else "+".join([f"w{c}" for c in range(col)])
        y = "0" if row == 0 else "+".join([f"h{r}" for r in range(row)])
        layout_positions.append(f"{x}_{y}")

    xstack = "".join(labeled) + f"xstack=inputs={n}:layout={'|'.join(layout_positions)}[out]"
    parts.append(xstack)
    return ";".join(parts), [str(cols), str(rows)]


def process_video(name: str, root: Path, methods: list[str], out_dir: Path,
                   tile_w: int, tile_h: int, overwrite: bool) -> None:
    out_path = out_dir / name
    if out_path.exists() and not overwrite:
        print(f"[skip] {name} (already exists)")
        return

    inputs = []
    for m in methods:
        inputs += ["-i", str(root / m / "videos" / name)]

    filter_complex, (cols, rows) = build_filter_complex(len(methods), methods, tile_w, tile_h)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-movflags", "+faststart",
        str(out_path),
    ]
    print(f"[run ] {name}  ({cols}x{rows} grid, {len(methods)} sources)")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="YOLOX_outputs", help="Root dir containing method folders")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, help="Method folder names, in grid order")
    parser.add_argument("--out", default=None, help="Output dir (default: <root>/video_cmp)")
    parser.add_argument("--tile-width", type=int, default=640, help="Width of each tile")
    parser.add_argument("--tile-height", type=int, default=360, help="Height of each tile")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--videos", nargs="*", default=None, help="Specific video file names to process (default: all common)")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH", file=sys.stderr)
        sys.exit(1)

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root / "video_cmp"
    out_dir.mkdir(parents=True, exist_ok=True)

    common = find_common_videos(root, args.methods)
    if not common:
        print("No common videos found across all method folders.", file=sys.stderr)
        sys.exit(1)

    targets = args.videos if args.videos else common
    missing = [v for v in targets if v not in common]
    if missing:
        print(f"Warning: not common to all methods, skipping: {missing}", file=sys.stderr)
        targets = [v for v in targets if v in common]

    print(f"Found {len(common)} common videos across {len(args.methods)} methods. Processing {len(targets)}.")
    for name in targets:
        process_video(name, root, args.methods, out_dir, args.tile_width, args.tile_height, args.overwrite)

    print(f"Done. Output in: {out_dir}")


if __name__ == "__main__":
    main()

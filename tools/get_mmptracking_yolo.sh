#!/usr/bin/env bash
# Download the MMPTracking YOLO shards (made by tools/build_mmptracking_yolo.py --upload-repo)
# from the Hugging Face Hub and unpack them into a ready-to-train YOLO dataset:
#
#   <out>/images/{train,val}/*.jpg
#   <out>/labels/{train,val}/*.txt
#   <out>/data.yaml            (path: set to the absolute <out>)
#
# Shards are downloaded and extracted one at a time, then deleted, so only ~1 shard of extra
# disk is needed. Finished shards are recorded in <out>/.done_shards: re-run to resume.
#
# Usage:
#   bash tools/get_mmptracking_yolo.sh                                   # defaults below
#   bash tools/get_mmptracking_yolo.sh -r user/repo -o /data/mmp_yolo
#   bash tools/get_mmptracking_yolo.sh -s val -i 'lobby_*'               # only val lobby scenes
#
# Private repo: export HF_TOKEN=hf_xxx (or run `hf auth login`) first.

set -euo pipefail

REPO="${REPO:-hungnguyen190204/MMPTracking-YOLO}"
PREFIX="${PREFIX:-yolo}"
OUT="${OUT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/datasets/MMPTracking/yolo}"
SPLITS="${SPLITS:-train val}"
INCLUDE="${INCLUDE:-*}"

usage() {
    cat <<EOF
Usage: $0 [-r repo] [-p prefix] [-o out_dir] [-s "train val"] [-i glob]
  -r  HF dataset repo id           (default: $REPO)
  -p  folder of the shards in repo (default: $PREFIX)
  -o  output directory             (default: $OUT)
  -s  splits to fetch              (default: "$SPLITS")
  -i  shard name glob, e.g. 'lobby_*' or '64am_*' (default: '$INCLUDE')
EOF
    exit 1
}

while getopts "r:p:o:s:i:h" opt; do
    case "$opt" in
        r) REPO="$OPTARG" ;;
        p) PREFIX="$OPTARG" ;;
        o) OUT="$OPTARG" ;;
        s) SPLITS="$OPTARG" ;;
        i) INCLUDE="$OPTARG" ;;
        *) usage ;;
    esac
done

PY="$(command -v python3 || command -v python)"
"$PY" -c "import huggingface_hub" 2>/dev/null || "$PY" -m pip install -q huggingface_hub

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
TMP="$OUT/_shards"
DONE="$OUT/.done_shards"
mkdir -p "$TMP"
touch "$DONE"

unpack() {  # unpack <zip> <dest>
    if command -v unzip >/dev/null; then
        unzip -q -o "$1" -d "$2"
    else
        "$PY" -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$1" "$2"
    fi
}

echo "Repo:   $REPO ($PREFIX/)"
echo "Output: $OUT"

for split in $SPLITS; do
    mapfile -t shards < <("$PY" - "$REPO" "$PREFIX/$split" "$INCLUDE" <<'EOF'
import fnmatch, sys
from huggingface_hub import HfApi
repo, folder, pattern = sys.argv[1:]
for f in sorted(HfApi().list_repo_files(repo, repo_type="dataset")):
    name = f.rsplit("/", 1)[-1]
    if f.startswith(folder + "/") and f.endswith(".zip") and fnmatch.fnmatch(name[:-4], pattern):
        print(f)
EOF
    )
    echo "== $split: ${#shards[@]} shards"
    [ "${#shards[@]}" -eq 0 ] && continue

    i=0
    for shard in "${shards[@]}"; do
        i=$((i + 1))
        if grep -qxF "$shard" "$DONE"; then
            echo "[$i/${#shards[@]}] $shard: done, skip"
            continue
        fi
        echo "[$i/${#shards[@]}] $shard"
        local_zip="$("$PY" - "$REPO" "$shard" "$TMP" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download(sys.argv[1], sys.argv[2], repo_type="dataset", local_dir=sys.argv[3]))
EOF
        )"
        unpack "$local_zip" "$OUT"
        rm -f "$local_zip"
        echo "$shard" >> "$DONE"
    done
done

rm -rf "$TMP"
cat > "$OUT/data.yaml" <<EOF
path: $OUT
train: images/train
val: images/val

names:
  0: person
EOF

for split in train val; do
    [ -d "$OUT/images/$split" ] || continue
    n_img=$(find "$OUT/images/$split" -name '*.jpg' | wc -l)
    n_lbl=$(find "$OUT/labels/$split" -name '*.txt' | wc -l)
    echo "$split: $n_img images, $n_lbl labels"
done
echo "Done. Train with: yolo detect train data=$OUT/data.yaml"

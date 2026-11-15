#!/usr/bin/env bash
# Download the SAM and ResNet50 checkpoints into ./checkpoints (or DIR).
# Usage: scripts/download_checkpoints.sh [--vit-h] [DIR]
#   --vit-h  also fetch SAM ViT-H (2.4 GB), used for the original evaluation
set -euo pipefail
urls=(
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
  "https://download.pytorch.org/models/resnet50-0676ba61.pth"
)
if [ "${1:-}" = "--vit-h" ]; then
  urls+=("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
  shift
fi
DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/checkpoints}"
mkdir -p "$DEST"
for url in "${urls[@]}"; do
  f="$DEST/$(basename "$url")"
  if [ -s "$f" ]; then
    echo "exists: $f"
  else
    echo "downloading: $url"
    curl -fL --retry 3 -o "$f.part" "$url" && mv "$f.part" "$f"
  fi
done

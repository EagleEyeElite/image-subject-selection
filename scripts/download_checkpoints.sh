#!/usr/bin/env bash
# Download the SAM and ResNet50 checkpoints into ./checkpoints (or $1).
set -euo pipefail
DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/checkpoints}"
mkdir -p "$DEST"
for url in \
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
  "https://download.pytorch.org/models/resnet50-0676ba61.pth"; do
  f="$DEST/$(basename "$url")"
  if [ -s "$f" ]; then echo "exists: $f"; else echo "downloading: $url"; curl -fL --retry 3 -o "$f.part" "$url" && mv "$f.part" "$f"; fi
done

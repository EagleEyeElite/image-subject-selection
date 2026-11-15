"""Evaluate on every suitable Pascal VOC 2012 image (single class, see
image_subject_selection.evaluation): all saliency methods plus the reference points.

Writes results/pascal_voc_<sam>_<timestamp>.csv (updated every 50 images, so an
interrupted run keeps what it has) and prints the mean per method.
About 2.5 hours for the 1 509 images with SAM ViT-B on one GPU.

Usage: uv run python scripts/benchmark.py [--sam vit_b] [--limit N]
"""

import argparse
from datetime import datetime

from image_subject_selection import load_resnet, load_sam
from image_subject_selection.evaluation import (
    RESULTS_DIR,
    Evaluator,
    default_metrics,
    default_xai_methods,
    label_table,
    load_pascal_voc,
    single_class_images,
)

COLUMNS = ["iou", "dice", "precision", "recall", "biou"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sam", choices=["vit_b", "vit_l", "vit_h"], default="vit_b")
    parser.add_argument("--limit", type=int, help="only the first N images (for a quick test)")
    args = parser.parse_args()

    sam, resnet = load_sam(args.sam), load_resnet()
    pascal = load_pascal_voc()
    indices = list(single_class_images(label_table(pascal))["dataset_idx"])[: args.limit]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"pascal_voc_{args.sam}_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
    print(f"{len(indices)} images -> {path}")

    evaluator = Evaluator(resnet, sam, default_xai_methods(resnet), default_metrics())
    results = evaluator.run(pascal, indices, checkpoint=path)
    print(results.groupby("xai_method")[COLUMNS].mean().sort_values("iou", ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()

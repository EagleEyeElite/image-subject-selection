"""Command line: segment images, write the mask and optionally the figures.

    image-subject-selection examples/falcon-heavy-boosters.jpeg -o out --figures
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np

from . import plotting
from .models import (
    IMAGENET_LABELS,
    load_image,
    load_resnet,
    load_sam,
    predict,
    to_tensor,
)
from .segmentation import MAX_AREA, SegmentationResult, mask_generator, segment
from .xai import GradCAM, LayerCAM, lrp_epsilon_plus, lrp_gamma_box, to_image_size

XAI_METHODS = {
    "layercam": lambda resnet: LayerCAM(resnet, resnet.layer4),
    "gradcam": lambda resnet: GradCAM(resnet, resnet.layer4),
    "lrp-ep": lrp_epsilon_plus,
    "lrp-gb": lrp_gamma_box,
}


def save_figures(
    image: np.ndarray, heatmap: np.ndarray, result: SegmentationResult, folder: Path, name: str, dpi: int = 90
) -> list[Path]:
    """Write ``<name>-overview/ranked/clusters/scores.png`` to ``folder``."""
    import matplotlib.pyplot as plt

    figures = {
        "overview": plotting.overview(image, heatmap, result),
        "ranked": plotting.ranked_masks(image, result, limit=16),
        "clusters": plotting.clusters(image, result),
        "scores": plotting.cluster_scores(result),
    }
    paths = []
    for kind, fig in figures.items():
        path = folder / f"{name}-{kind}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="image-subject-selection",
        description="Segment the object ResNet50 sees: SAM segments everything, a saliency map selects.",
    )
    parser.add_argument("images", nargs="+", type=Path, help="input images")
    parser.add_argument("-o", "--output", type=Path, default=Path("."), help="output folder (default: .)")
    parser.add_argument("--xai", choices=XAI_METHODS, default="layercam", help="saliency method (default: layercam)")
    parser.add_argument("--sam", choices=["vit_b", "vit_l", "vit_h"], default="vit_b", help="SAM model (default: vit_b)")
    parser.add_argument("--max-area", type=float, default=MAX_AREA,
                        help=f"clusters larger than this percentage of the image are background (default: {MAX_AREA:g})")
    parser.add_argument("--figures", action="store_true", help="also write the figures of every pipeline step")
    parser.add_argument("--dpi", type=int, default=90, help="resolution of the figures (default: 90)")
    args = parser.parse_args(argv)

    matplotlib.use("Agg")
    sam, resnet = load_sam(args.sam), load_resnet()
    generator = mask_generator(sam)
    method = XAI_METHODS[args.xai](resnet)
    device = next(resnet.parameters()).device
    args.output.mkdir(parents=True, exist_ok=True)

    for path in args.images:
        image = load_image(path)
        img_tensor = to_tensor(image).unsqueeze(0).to(device)
        class_idx = predict(resnet, img_tensor)
        heatmap = to_image_size(method(img_tensor, class_idx), image.shape)
        result = segment(image, heatmap, generator, max_area=args.max_area)

        print(f"{path}: {IMAGENET_LABELS[class_idx]}, {len(result.selected)} of {len(result.clusters)} clusters, "
              f"{result.mask.mean():.1%} of the image")
        print(plotting.cluster_table(result))
        written = [args.output / f"{path.stem}-mask.png"]
        cv2.imwrite(str(written[0]), result.mask.astype(np.uint8) * 255)
        if args.figures:
            written += save_figures(image, heatmap, result, args.output, path.stem, args.dpi)
        for p in written:
            print("wrote", p)


if __name__ == "__main__":
    main()

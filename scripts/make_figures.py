"""Regenerate the README figures in docs/images from examples/falcon-heavy-boosters.jpeg.

Usage: uv run python scripts/make_figures.py
"""

import matplotlib

matplotlib.use("Agg")

from saliency_sam import (
    load_image,
    load_resnet,
    load_sam,
    mask_generator,
    plotting,
    predict,
    segment,
    to_tensor,
)
from saliency_sam.models import IMAGENET_LABELS, PROJECT_ROOT
from saliency_sam.xai import LayerCAM, to_image_size

OUT = PROJECT_ROOT / "docs" / "images"


def main():
    image = load_image(PROJECT_ROOT / "examples" / "falcon-heavy-boosters.jpeg")
    sam, resnet = load_sam("vit_b"), load_resnet()

    img_tensor = to_tensor(image).unsqueeze(0).to(next(resnet.parameters()).device)
    class_idx = predict(resnet, img_tensor)
    heatmap = to_image_size(LayerCAM(resnet, resnet.layer4)(img_tensor, class_idx), image.shape)
    result = segment(image, heatmap, mask_generator(sam))

    print("ResNet50 prediction:", IMAGENET_LABELS[class_idx])
    print(plotting.cluster_table(result))

    figures = {
        "falcon-overview": plotting.overview(image, heatmap, result),
        "falcon-ranked": plotting.ranked_masks(image, result, limit=16),
        "falcon-clusters": plotting.clusters(image, result),
        "falcon-scores": plotting.cluster_scores(result),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        fig.savefig(OUT / f"{name}.png", dpi=90, bbox_inches="tight")
        print("wrote", OUT / f"{name}.png")


if __name__ == "__main__":
    main()

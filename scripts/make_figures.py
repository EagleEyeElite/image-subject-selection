"""Regenerate the README figures in docs/images: the pipeline steps on the
example image (the same as `image-subject-selection examples/falcon-heavy-boosters.jpeg --figures`).

Usage: uv run python scripts/make_figures.py
"""

import matplotlib

matplotlib.use("Agg")

from image_subject_selection import (
    load_image,
    load_resnet,
    load_sam,
    mask_generator,
    plotting,
    predict,
    segment,
    to_tensor,
)
from image_subject_selection.cli import save_figures
from image_subject_selection.models import IMAGENET_LABELS, PROJECT_ROOT
from image_subject_selection.xai import LayerCAM, to_image_size

OUT = PROJECT_ROOT / "docs" / "images"


def main():
    sam, resnet = load_sam("vit_b"), load_resnet()
    image = load_image(PROJECT_ROOT / "examples" / "falcon-heavy-boosters.jpeg")
    img_tensor = to_tensor(image).unsqueeze(0).to(next(resnet.parameters()).device)
    class_idx = predict(resnet, img_tensor)
    heatmap = to_image_size(LayerCAM(resnet, resnet.layer4)(img_tensor, class_idx), image.shape)
    result = segment(image, heatmap, mask_generator(sam))

    print("ResNet50 prediction:", IMAGENET_LABELS[class_idx])
    print(plotting.cluster_table(result))
    OUT.mkdir(parents=True, exist_ok=True)
    for path in save_figures(image, heatmap, result, OUT, "falcon-heavy-boosters"):
        print("wrote", path)


if __name__ == "__main__":
    main()

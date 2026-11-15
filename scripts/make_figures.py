"""Regenerate the README figures in examples/figures: the pipeline steps on the
Falcon Heavy example (the same as `image-subject-selection examples/falcon-heavy-boosters.jpeg --figures`)
and the overviews of the corgi and seal examples.

Usage: uv run python scripts/make_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

OUT = PROJECT_ROOT / "examples" / "figures"


def run(sam, resnet, filename):
    image = load_image(PROJECT_ROOT / "examples" / filename)
    img_tensor = to_tensor(image).unsqueeze(0).to(next(resnet.parameters()).device)
    class_idx = predict(resnet, img_tensor)
    heatmap = to_image_size(LayerCAM(resnet, resnet.layer4)(img_tensor, class_idx), image.shape)
    result = segment(image, heatmap, mask_generator(sam))
    print(f"{filename}: ResNet50 prediction:", IMAGENET_LABELS[class_idx])
    print(plotting.cluster_table(result))
    return image, heatmap, result


def main():
    sam, resnet = load_sam("vit_b"), load_resnet()
    OUT.mkdir(parents=True, exist_ok=True)

    for path in save_figures(*run(sam, resnet, "falcon-heavy-boosters.jpeg"), OUT, "falcon-heavy-boosters"):
        print("wrote", path)

    for name in ["corgi", "seal"]:
        fig = plotting.overview(*run(sam, resnet, f"{name}.jpg"))
        path = OUT / f"{name}-overview.png"
        fig.savefig(path, dpi=90, bbox_inches="tight")
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()

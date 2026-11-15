"""Regenerate the README figures in docs/images.

- falcon-heavy-boosters-*.png: the pipeline steps on the example image
  (the same as `image-subject-selection examples/falcon-heavy-boosters.jpeg --figures`)
- voc-comparison.png: GradCAM vs. LRP on the Pascal VOC sample of the
  evaluation notebook (same seed, same 5 images as the README table);
  skipped without datasets/VOCSegmentation

Usage: uv run python scripts/make_figures.py
"""

import random

import matplotlib

matplotlib.use("Agg")

import numpy as np

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
from image_subject_selection.evaluation import (
    PASCAL_ROOT,
    class_names,
    label_table,
    load_pascal_voc,
    object_mask,
    single_class_images,
)
from image_subject_selection.metrics import IoU
from image_subject_selection.models import IMAGENET_LABELS, PROJECT_ROOT
from image_subject_selection.xai import (
    GradCAM,
    LayerCAM,
    lrp_epsilon_plus,
    lrp_gamma_box,
    to_image_size,
)

OUT = PROJECT_ROOT / "docs" / "images"
SEED, N_SAMPLES = 123, 5  # as in notebooks/evaluation.ipynb


def falcon(sam, resnet):
    image = load_image(PROJECT_ROOT / "examples" / "falcon-heavy-boosters.jpeg")
    img_tensor = to_tensor(image).unsqueeze(0).to(next(resnet.parameters()).device)
    class_idx = predict(resnet, img_tensor)
    heatmap = to_image_size(LayerCAM(resnet, resnet.layer4)(img_tensor, class_idx), image.shape)
    result = segment(image, heatmap, mask_generator(sam))
    print("ResNet50 prediction:", IMAGENET_LABELS[class_idx])
    print(plotting.cluster_table(result))
    return save_figures(image, heatmap, result, OUT, "falcon-heavy-boosters")


def voc_comparison(sam, resnet):
    pascal = load_pascal_voc()
    candidates = single_class_images(label_table(pascal))
    random.seed(SEED)
    sample = random.sample(list(candidates["dataset_idx"]), N_SAMPLES)

    methods = {
        "GradCAM": GradCAM(resnet, resnet.layer4),
        "LRP Epsilon-Plus": lrp_epsilon_plus(resnet),
        "LRP Epsilon-Gamma-Box": lrp_gamma_box(resnet),
    }
    generator, iou = mask_generator(sam), IoU()
    device = next(resnet.parameters()).device
    rows = []
    for idx in sample:
        pil_image, target = pascal[idx]
        image = np.array(pil_image.convert("RGB"))
        truth = object_mask(target)
        img_tensor = to_tensor(image).unsqueeze(0).to(device)
        class_idx = predict(resnet, img_tensor)
        selections = {}
        for name, method in methods.items():
            heatmap = to_image_size(method(img_tensor, class_idx), image.shape)
            mask = segment(image, heatmap, generator).mask
            selections[name] = (mask, iou(mask.astype(np.uint8), truth))
        rows.append({
            "image": image,
            "ground_truth": truth,
            "label": f"{class_names(target)} / {IMAGENET_LABELS[class_idx]}",
            "selections": selections,
        })
        print(idx, {name: round(score, 3) for name, (_, score) in selections.items()})

    fig = plotting.method_comparison(rows)
    path = OUT / "voc-comparison.png"
    fig.savefig(path, dpi=80, bbox_inches="tight")
    return [path]


def main():
    sam, resnet = load_sam("vit_b"), load_resnet()
    OUT.mkdir(parents=True, exist_ok=True)
    written = falcon(sam, resnet)
    if PASCAL_ROOT.is_dir():
        written += voc_comparison(sam, resnet)
    else:
        print(f"{PASCAL_ROOT} missing, voc-comparison.png skipped")
    for path in written:
        print("wrote", path)


if __name__ == "__main__":
    main()

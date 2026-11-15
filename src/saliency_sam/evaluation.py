"""Evaluation on Pascal VOC 2012: segment with every XAI method, compare to ground truth."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torchvision
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from tqdm.auto import tqdm

from .models import IMAGENET_LABELS, PROJECT_ROOT, predict, to_tensor
from .segmentation import mask_generator, segment
from .xai import to_image_size

PASCAL_ROOT = PROJECT_ROOT / "datasets" / "VOCSegmentation"
RESULTS_DIR = PROJECT_ROOT / "results"

PASCAL_LABELS = {
    1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle",
    6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow",
    11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person",
    16: "potted plant", 17: "sheep", 18: "sofa", 19: "train", 20: "tv/monitor",
}

# Classes without a clear ImageNet counterpart or usually only partly visible.
EXCLUDED_CLASSES = ["person", "tv/monitor", "sofa", "potted plant", "diningtable", "chair", "bottle"]

VOID = 255  # object borders and ambiguous pixels in the Pascal masks


class SquareCrop:
    """Center crop a PIL image to a square, optionally resize it."""

    def __init__(self, size: int | None = None, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        if self.size is not None:
            img = img.resize((self.size, self.size), self.interpolation)
        return img


def load_pascal_voc(root: str | Path = PASCAL_ROOT, image_sets=("train", "val"), download: bool = False) -> Dataset:
    """Pascal VOC 2012 segmentation (train + val), images and masks center-cropped to squares."""
    crop = SquareCrop(interpolation=Image.NEAREST)
    return ConcatDataset([
        torchvision.datasets.VOCSegmentation(
            root=str(root), year="2012", image_set=image_set, download=download,
            transform=crop, target_transform=crop,
        )
        for image_set in image_sets
    ])


def class_ids(target) -> np.ndarray:
    ids = np.unique(np.asarray(target))
    return ids[(ids != 0) & (ids != VOID)]


def class_names(target) -> str:
    return ", ".join(PASCAL_LABELS.get(int(i), "unknown") for i in class_ids(target)) or "unknown"


def object_mask(target) -> np.ndarray:
    """Foreground of a Pascal mask as uint8 0/1 (background and void are 0)."""
    target = np.asarray(target)
    return ((target != 0) & (target != VOID)).astype(np.uint8)


def label_table(dataset: Dataset) -> pd.DataFrame:
    """One row per image: dataset index, class ids and names. Reads every mask once."""
    rows = []
    for idx in tqdm(range(len(dataset)), desc="reading masks"):
        ids = class_ids(dataset[idx][1])
        rows.append({
            "dataset_idx": idx,
            "labels": ids,
            "label_count": len(ids),
            "label_names": [PASCAL_LABELS[int(i)] for i in ids],
        })
    return pd.DataFrame(rows)


def single_class_images(table: pd.DataFrame, excluded=EXCLUDED_CLASSES) -> pd.DataFrame:
    """Rows of images that contain exactly one class, and not an excluded one."""
    keep = (table["label_count"] == 1) & ~table["label_names"].apply(lambda names: names[0] in excluded if names else True)
    return table[keep]


class Evaluator:
    """Runs the pipeline with several XAI methods per image and collects metrics.

    ``xai_methods`` maps a name to a method from ``saliency_sam.xai``;
    ``metrics`` maps a name to a metric from ``saliency_sam.metrics``.
    """

    def __init__(self, resnet, sam, xai_methods: dict, metrics: dict, labels=IMAGENET_LABELS, **generator_settings):
        self.resnet = resnet
        self.generator = mask_generator(sam, **generator_settings)
        self.xai_methods = xai_methods
        self.metrics = metrics
        self.labels = labels
        self.rows = []

    def __call__(self, image: Image.Image, target, dataset_idx: int) -> list[dict]:
        """Evaluate one image; returns one entry per XAI method with the masks for plotting."""
        image_rgb = np.array(image.convert("RGB"))
        truth = object_mask(target)
        device = next(self.resnet.parameters()).device
        img_tensor = to_tensor(image_rgb).unsqueeze(0).to(device)
        class_idx = predict(self.resnet, img_tensor)

        outputs = []
        for name, method in self.xai_methods.items():
            heatmap = to_image_size(method(img_tensor, class_idx), image_rgb.shape)
            result = segment(image_rgb, heatmap, self.generator)
            prediction = result.mask.astype(np.uint8)
            self.rows.append({
                "dataset_idx": dataset_idx,
                "xai_method": name,
                "true_label": class_names(target),
                "pred_label": self.labels[class_idx],
                **{metric_name: metric(prediction, truth) for metric_name, metric in self.metrics.items()},
            })
            outputs.append({
                "xai_method": name,
                "image": image_rgb,
                "heatmap": heatmap,
                "mask": result.mask,
                "ground_truth": truth,
                "true_label": class_names(target),
                "pred_label": self.labels[class_idx],
            })
        return outputs

    def run(self, dataset: Dataset, indices) -> pd.DataFrame:
        for idx in tqdm(indices):
            image, target = dataset[idx]
            self(image, target, idx)
        return self.results

    @property
    def results(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def save_results(df: pd.DataFrame, name: str, folder: str | Path = RESULTS_DIR) -> Path:
    """Write ``<folder>/<name>_<timestamp>.csv``."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
    df.to_csv(path, index=False)
    return path

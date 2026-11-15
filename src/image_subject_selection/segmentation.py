"""The pipeline: rank SAM's masks by the saliency map, cluster them, pick by elbow."""

from dataclasses import dataclass

import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator

# Dense settings from SAM's example notebook: crops find small objects too.
GENERATOR_SETTINGS = dict(
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # post-processing with OpenCV
)


@dataclass
class Cluster:
    mask: np.ndarray  # bool (H, W)
    score: float  # weighted IoU with the saliency map
    area: float  # percent of the image


@dataclass
class SegmentationResult:
    masks: list[dict]  # SAM's masks, best score first
    mask_scores: list[float]
    clusters: list[Cluster]  # best score first
    candidates: list[Cluster]  # clusters not larger than max_area, best score first
    selected: list[Cluster]  # the candidates before the elbow
    mask: np.ndarray  # bool (H, W): union of the selected clusters


def mask_generator(sam, **settings) -> SamAutomaticMaskGenerator:
    """SAM's automatic mask generator with GENERATOR_SETTINGS, overridable."""
    return SamAutomaticMaskGenerator(model=sam, **{**GENERATOR_SETTINGS, **settings})


def weighted_iou(mask: torch.Tensor, heatmap: torch.Tensor) -> float:
    """Soft IoU of a binary mask and a [0, 1] heatmap, times the mask's mean relevance.

    The second factor penalizes large masks that cover the salient region
    but mostly consist of irrelevant pixels (e.g. the background).
    """
    intersection = (mask * heatmap).sum()
    union = (mask + heatmap - mask * heatmap).sum()
    if union == 0 or mask.sum() == 0:
        return 0.0
    return (intersection / union * intersection / mask.sum()).item()


def rank_masks(masks: list[dict], heatmap: np.ndarray) -> tuple[list[dict], list[float]]:
    """SAM masks sorted by weighted IoU with the heatmap, best first."""
    heatmap_t = torch.as_tensor(heatmap, dtype=torch.float32, device=_device())
    scores = [weighted_iou(_tensor(m["segmentation"]), heatmap_t) for m in masks]
    order = sorted(range(len(masks)), key=scores.__getitem__, reverse=True)
    return [masks[i] for i in order], [scores[i] for i in order]


def cluster_masks(masks: list[dict], heatmap: np.ndarray, overlap_threshold: float = 0.2) -> list[Cluster]:
    """Merge overlapping masks into clusters, sorted by weighted IoU.

    Going through ``masks`` in order, each unused mask starts a cluster and
    absorbs every later unused mask that shares more than
    ``overlap_threshold`` of its own or of the seed's area.
    """
    heatmap_t = torch.as_tensor(heatmap, dtype=torch.float32, device=_device())
    tensors = [_tensor(m["segmentation"]) for m in masks]
    areas = [t.sum() for t in tensors]

    merged, used = [], set()
    for i, seed in enumerate(tensors):
        if i in used:
            continue
        used.add(i)
        combined = seed.clone()
        for j in range(i + 1, len(tensors)):
            if j in used:
                continue
            overlap = (seed * tensors[j]).sum()
            if overlap > 0 and (overlap / areas[i] > overlap_threshold or overlap / areas[j] > overlap_threshold):
                combined = torch.maximum(combined, tensors[j])
                used.add(j)
        merged.append(combined)

    clusters = [
        Cluster(
            mask=c.bool().cpu().numpy(),
            score=weighted_iou(c, heatmap_t),
            area=100 * c.mean().item(),
        )
        for c in merged
    ]
    return sorted(clusters, key=lambda c: c.score, reverse=True)


def elbow(scores: list[float]) -> int:
    """Index of the elbow of a descending curve: the point farthest from the
    line between its first and last point. At least 1."""
    if len(scores) < 3:
        return len(scores)
    points = np.column_stack([np.arange(len(scores)), scores])
    line = points[-1] - points[0]
    line /= np.linalg.norm(line)
    offsets = points - points[0]
    distances = np.linalg.norm(offsets - np.outer(offsets @ line, line), axis=1)
    return max(int(distances.argmax()), 1)


def combine(clusters: list[Cluster], shape: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=bool)
    for cluster in clusters:
        mask |= cluster.mask
    return mask


def segment(
    image: np.ndarray, heatmap: np.ndarray, generator: SamAutomaticMaskGenerator, max_area: float = 50.0
) -> SegmentationResult:
    """Run the whole pipeline on an RGB image and its heatmap (same size, [0, 1])."""
    return select(generator.generate(image), heatmap, max_area)


def select(masks: list[dict], heatmap: np.ndarray, max_area: float = 50.0) -> SegmentationResult:
    """Pick from SAM's masks of an image with the heatmap; ``segment`` without the SAM pass,
    so several heatmaps can share one set of masks.

    Clusters covering more than ``max_area`` percent of the image are background
    (sky, ground, walls) and never selected: with a weak heatmap around the
    object, their sheer size can still earn them a high score.
    """
    masks, mask_scores = rank_masks(masks, heatmap)
    clusters = cluster_masks(masks, heatmap)
    candidates = [c for c in clusters if c.area <= max_area]
    selected = candidates[:elbow([c.score for c in candidates])]
    return SegmentationResult(masks, mask_scores, clusters, candidates, selected, combine(selected, heatmap.shape))


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tensor(mask: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(mask, dtype=torch.float32, device=_device())

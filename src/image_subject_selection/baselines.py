"""Reference points for the evaluation, computed from the same SAM masks.

- ``largest_mask`` and ``center_mask`` use no saliency map: if the pipeline
  does not beat them, the classifier's explanation adds nothing.
- ``best_masks`` looks at the ground truth: about the best any selection of
  SAM's masks can reach, so it separates what SAM misses from what the
  selection misses.
"""

import numpy as np


def largest_mask(masks: list[dict], shape: tuple[int, ...], max_area: float = 50.0) -> np.ndarray:
    """The largest SAM mask that is not background (at most ``max_area`` percent of the image)."""
    limit = max_area / 100 * shape[0] * shape[1]
    candidates = [m for m in masks if m["area"] <= limit]
    if not candidates:
        return np.zeros(shape[:2], dtype=bool)
    return max(candidates, key=lambda m: m["area"])["segmentation"]


def center_mask(masks: list[dict], shape: tuple[int, ...], max_area: float = 50.0) -> np.ndarray:
    """The largest non-background SAM mask containing the image center; objects
    in photos tend to be centered. Falls back to ``largest_mask``."""
    cy, cx = shape[0] // 2, shape[1] // 2
    limit = max_area / 100 * shape[0] * shape[1]
    candidates = [m for m in masks if m["area"] <= limit and m["segmentation"][cy, cx]]
    if not candidates:
        return largest_mask(masks, shape, max_area)
    return max(candidates, key=lambda m: m["area"])["segmentation"]


def best_masks(masks: list[dict], truth: np.ndarray) -> np.ndarray:
    """Union of SAM masks chosen with the ground truth to maximize IoU.

    Greedy: masks in order of the fraction of their pixels inside the object,
    each added if it raises the IoU. Choosing the best union exactly is
    combinatorial; the greedy choice is a close lower bound of that optimum.
    """
    truth = truth.astype(bool)
    union = np.zeros_like(truth)
    if not truth.any():
        return union
    inside = [(m["segmentation"] & truth).sum() / max(m["area"], 1) for m in masks]
    best_iou = 0.0
    for i in np.argsort(inside)[::-1]:
        if inside[i] == 0:
            break
        candidate = union | masks[i]["segmentation"]
        iou = (candidate & truth).sum() / (candidate | truth).sum()
        if iou > best_iou:
            union, best_iou = candidate, iou
    return union

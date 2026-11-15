"""Figures for each step of the pipeline. Every function returns its figure."""

import math

import matplotlib.pyplot as plt
import numpy as np

from .segmentation import SegmentationResult

RED = (1.0, 0.0, 0.0)


def overlay(image: np.ndarray, mask: np.ndarray, color=RED, alpha: float = 0.5) -> np.ndarray:
    """RGB image with ``mask`` tinted in ``color``."""
    out = image.astype(np.float32) / 255
    out[mask] = (1 - alpha) * out[mask] + alpha * np.asarray(color)
    return out


def all_masks(image: np.ndarray, masks: list[dict], alpha: float = 0.6, seed: int = 0) -> np.ndarray:
    """All SAM masks, each in a random color, largest first so small ones stay visible."""
    rng = np.random.default_rng(seed)
    out = image.astype(np.float32) / 255
    for m in sorted(masks, key=lambda m: m["area"], reverse=True):
        seg = m["segmentation"]
        out[seg] = (1 - alpha) * out[seg] + alpha * rng.random(3)
    return out


def overview(image: np.ndarray, heatmap: np.ndarray, result: SegmentationResult, title: str | None = None):
    """Image | saliency map | all SAM masks | selection: the pipeline in one row."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[1].imshow(image)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.5)
    axes[1].set_title("Saliency map")
    axes[2].imshow(all_masks(image, result.masks))
    axes[2].set_title(f"SAM: {len(result.masks)} masks")
    axes[3].imshow(overlay(image, result.mask))
    axes[3].set_title(f"Selection: {len(result.selected)} of {len(result.clusters)} clusters")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def ranked_masks(image: np.ndarray, result: SegmentationResult, cols: int = 8, limit: int | None = 24):
    """SAM's masks, best weighted IoU first."""
    masks = result.masks[:limit]
    scores = result.mask_scores[:limit]
    return _grid(
        [overlay(image, m["segmentation"], alpha=0.7) for m in masks],
        [f"#{i + 1}  {s:.4f}" for i, s in enumerate(scores)],
        cols,
    )


def clusters(image: np.ndarray, result: SegmentationResult, cols: int = 6):
    """Clusters, best weighted IoU first; selected ones in bold red, too large ones in gray."""
    selected = {id(c) for c in result.selected}
    candidates = {id(c) for c in result.candidates}
    titles = [f"{c.score:.4f} | {c.area:.1f}%" for c in result.clusters]
    fig = _grid([overlay(image, c.mask) for c in result.clusters], titles, cols)
    for ax, c in zip(fig.axes, result.clusters):
        if id(c) in selected:
            ax.title.set_fontweight("bold")
            ax.title.set_color("darkred")
        elif id(c) not in candidates:
            ax.set_title(f"{ax.get_title()} (too large)", fontsize=9, color="gray")
    return fig


def cluster_scores(result: SegmentationResult):
    """Score curve of the candidate clusters (not too large) with the elbow cut."""
    scores = [c.score for c in result.candidates]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(1, len(scores) + 1), scores, marker="o")
    ax.axvline(len(result.selected) + 0.5, color="darkred", linestyle="--", label="elbow: selection ends")
    excluded = len(result.clusters) - len(result.candidates)
    if excluded:
        ax.set_title(f"{excluded} cluster(s) above the area limit left out", fontsize=9)
    ax.set_xlabel("Candidate rank")
    ax.set_ylabel("Weighted IoU")
    ax.legend()
    fig.tight_layout()
    return fig


def cluster_table(result: SegmentationResult) -> str:
    """One line per cluster; * selected, - too large to be a candidate."""
    chosen = {id(c) for c in result.selected}
    candidates = {id(c) for c in result.candidates}

    def mark(c):
        return "*" if id(c) in chosen else " " if id(c) in candidates else "-"

    return "\n".join(
        f"{mark(c)} cluster {i + 1:2d}: weighted IoU {c.score:.6f}, area {c.area:5.2f}%"
        for i, c in enumerate(result.clusters)
    )


def evaluation_grid(samples: list[list[dict]]):
    """One row per image: the image, then per XAI method the selection (gray),
    the saliency map (hot) and the ground truth outline (green)."""
    rows, cols = len(samples), len(samples[0]) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows), squeeze=False)
    for r, outputs in enumerate(samples):
        first = outputs[0]
        axes[r, 0].imshow(first["image"])
        axes[r, 0].text(-0.08, 0.5, f"True: {first['true_label']}\nPred.: {first['pred_label']}",
                        rotation=90, ha="center", va="center", transform=axes[r, 0].transAxes)
        for c, out in enumerate(outputs, start=1):
            axes[r, c].imshow(out["mask"], cmap="gray")
            axes[r, c].imshow(out["heatmap"], cmap="hot", alpha=0.5)
            axes[r, c].contour(out["ground_truth"], levels=[0.5], colors="lime", linewidths=1)
            if r == 0:
                axes[r, c].set_title(out["xai_method"])
    if rows:
        axes[0, 0].set_title("Image")
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    return fig


def _grid(images: list[np.ndarray], titles: list[str], cols: int):
    rows = max(1, math.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, img, title in zip(axes.flat, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig

"""Segmentation metrics: predicted mask vs. ground truth mask (2D arrays).

Non-uint8 predictions are binarized with ``> threshold`` (or ``>=``).
"""

import logging

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from scipy.spatial.distance import cdist, directed_hausdorff

EPSILON = 1e-10


def extract_boundary(mask, dilation_radius):
    """
    Extract the boundary of a binary mask using morphological dilation.

    Args:
    - mask (numpy array): Binary segmentation mask (values 0 or 1).
    - dilation_radius (int): Radius of dilation (default is 1).

    Returns:
    - boundary (numpy array): Binary boundary mask.
    """

    structure = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1))
    dilated_mask = binary_dilation(mask, structure=structure)

    boundary = np.logical_xor(dilated_mask, mask)
    return boundary


class IoU:
    def __init__(self, threshold=0):
        self.threshold = threshold

    def __call__(self, pred, true_mask):
        if pred.dtype != np.uint8:
            pred = pred > self.threshold
        intersection = np.logical_and(pred, true_mask).sum()
        union = np.logical_or(pred, true_mask).sum()
        return intersection / (union + EPSILON)


class DiceCoefficient:
    def __init__(self, threshold=0):
        self.threshold = threshold

    def __call__(self, pred, true_mask):
        if pred.dtype != np.uint8:
            pred = pred > self.threshold
        intersection = np.logical_and(pred, true_mask).sum()
        return 2 * intersection / ((pred.sum() + EPSILON) + true_mask.sum())


class Precision:
    def __init__(self, threshold=0):
        self.threshold = threshold

    def __call__(self, pred, true_mask):
        if pred.dtype != np.uint8:
            pred = pred > self.threshold
        tp = np.logical_and(pred, true_mask).sum()
        fp = np.logical_and(pred, np.logical_not(true_mask)).sum()
        return tp / (tp + fp + EPSILON)


class Recall:
    def __init__(self, threshold=0):
        self.threshold = threshold

    def __call__(self, pred, true_mask):
        if pred.dtype != np.uint8:
            pred = pred > self.threshold
        tp = np.logical_and(pred, true_mask).sum()
        fn = np.logical_and(np.logical_not(pred), true_mask).sum()
        return tp / (tp + fn + EPSILON)


class F1Score:
    def __init__(self, threshold=0):
        self.threshold = threshold

    def __call__(self, pred, true_mask):
        prec = Precision(self.threshold)(pred, true_mask)
        rec = Recall(self.threshold)(pred, true_mask)
        return 2 * (prec * rec) / (prec + rec + EPSILON)


class HausdorffDistance:
    def __init__(self, threshold=0):
        self.threshold = threshold

    def __call__(self, predicted_mask, ground_truth_mask):
        if predicted_mask.shape != ground_truth_mask.shape:
            raise ValueError(
                "Predicted mask and ground truth mask must have the same dimensions."
            )

        if predicted_mask.dtype != np.uint8:
            predicted_mask = (predicted_mask >= self.threshold).astype(np.uint8)
        if ground_truth_mask.dtype != np.uint8:
            ground_truth_mask = (ground_truth_mask >= self.threshold).astype(np.uint8)

        predicted_boundary = np.column_stack(np.where(predicted_mask > 0))
        ground_truth_boundary = np.column_stack(np.where(ground_truth_mask > 0))

        hd_pred_to_gt = directed_hausdorff(predicted_boundary, ground_truth_boundary)[0]
        hd_gt_to_pred = directed_hausdorff(ground_truth_boundary, predicted_boundary)[0]

        return max(hd_pred_to_gt, hd_gt_to_pred)


class AverageSurfaceDistance:
    def __init__(self, threshold=0):
        self.threshold = threshold

    def __call__(self, predicted_mask, ground_truth_mask):
        if predicted_mask.shape != ground_truth_mask.shape:
            raise ValueError(
                "Predicted mask and ground truth mask must have the same dimensions."
            )

        if predicted_mask.dtype != np.uint8:
            predicted_mask = (predicted_mask >= self.threshold).astype(np.uint8)
        if ground_truth_mask.dtype != np.uint8:
            ground_truth_mask = (ground_truth_mask >= self.threshold).astype(np.uint8)

        pred_boundary = predicted_mask - binary_erosion(predicted_mask)
        gt_boundary = ground_truth_mask - binary_erosion(ground_truth_mask)

        pred_to_gt_distances = distance_transform_edt(1 - gt_boundary)
        gt_to_pred_distances = distance_transform_edt(1 - pred_boundary)

        pred_to_gt_surface_dist = pred_to_gt_distances[pred_boundary > 0]
        gt_to_pred_surface_dist = gt_to_pred_distances[gt_boundary > 0]

        return (np.mean(pred_to_gt_surface_dist) + np.mean(gt_to_pred_surface_dist)) / 2


class AverageSymmetricSurfaceDistance:
    def __init__(self, dilation_radius, threshold=0):
        self.dilation_radius = dilation_radius
        self.threshold = threshold

    def __call__(self, predicted_mask, ground_truth_mask):
        if predicted_mask.shape != ground_truth_mask.shape:
            raise ValueError(
                "Predicted heatmap and ground truth mask must have the same dimensions."
            )

        if predicted_mask.dtype != np.uint8:
            predicted_mask = (predicted_mask >= self.threshold).astype(np.uint8)
        if ground_truth_mask.dtype != np.uint8:
            ground_truth_mask = (ground_truth_mask >= self.threshold).astype(np.uint8)

        predicted_boundary = extract_boundary(predicted_mask, self.dilation_radius)
        ground_truth_boundary = extract_boundary(
            ground_truth_mask, self.dilation_radius
        )

        pred_boundary_coords = np.column_stack(np.where(predicted_boundary > 0))
        gt_boundary_coords = np.column_stack(np.where(ground_truth_boundary > 0))

        if len(pred_boundary_coords) == 0 or len(gt_boundary_coords) == 0:
            logging.warning(f"Pred boundary coords: {len(pred_boundary_coords)}, GT boundary coords: {len(gt_boundary_coords)}")
            return 0.0

        distances_pred_to_gt = cdist(pred_boundary_coords, gt_boundary_coords).min(
            axis=1
        )
        distances_gt_to_pred = cdist(gt_boundary_coords, pred_boundary_coords).min(
            axis=1
        )

        return(np.sum(distances_pred_to_gt) + np.sum(distances_gt_to_pred)) / (
            len(pred_boundary_coords) + len(gt_boundary_coords) + EPSILON
        )


class BoundaryIoU:
    def __init__(self, dilation_radius=1, threshold=0):
        self.dilation_radius = dilation_radius
        self.threshold = threshold

    def __call__(self, predicted_mask, ground_truth_mask):
        if ground_truth_mask.dtype != np.uint8:
            ground_truth_mask = (ground_truth_mask >= self.threshold).astype(np.uint8)
        if predicted_mask.dtype != np.uint8:
            predicted_mask = (predicted_mask >= self.threshold).astype(np.uint8)

        predicted_boundary = extract_boundary(predicted_mask, self.dilation_radius)
        ground_truth_boundary = extract_boundary(
            ground_truth_mask, self.dilation_radius
        )

        boundary_intersection = np.sum(predicted_boundary & ground_truth_boundary)
        boundary_union = np.sum(predicted_boundary | ground_truth_boundary)

        if boundary_union == 0:
            return 1.0
        else:
            return boundary_intersection / (boundary_union + EPSILON)


class NormalizedSurfaceDistance:
    def __init__(self, spacing=(1.0, 1.0), threshold=0):
        """
        Initialize the Normalized Surface Distance (NSD) calculator.

        Parameters:
        - spacing: tuple, voxel spacing in each dimension.
        """
        self.spacing = spacing
        self.threshold = threshold

    def __call__(self, pred_mask, mask):
        """
        Calculate the Normalized Surface Distance (NSD) between two binary masks.

        Parameters:
        - pred_mask: np.ndarray, predicted binary mask.
        - mask: np.ndarray, ground truth binary mask.

        Returns:
        - nsd: float, the normalized surface distance.
        """

        if pred_mask.dtype != np.uint8:
            pred_mask = (pred_mask >= self.threshold).astype(np.uint8)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        dt_mask = distance_transform_edt(mask == 0, sampling=self.spacing)
        dt_pred_mask = distance_transform_edt(pred_mask == 0, sampling=self.spacing)

        surface_mask = np.logical_xor(
            mask, np.logical_and(mask, np.roll(mask, 1, axis=0))
        )
        surface_mask |= np.logical_xor(
            mask, np.logical_and(mask, np.roll(mask, 1, axis=1))
        )

        surface_pred_mask = np.logical_xor(
            pred_mask, np.logical_and(pred_mask, np.roll(pred_mask, 1, axis=0))
        )
        surface_pred_mask |= np.logical_xor(
            pred_mask, np.logical_and(pred_mask, np.roll(pred_mask, 1, axis=1))
        )

        distances_mask_to_pred = dt_pred_mask[surface_mask]
        distances_pred_to_mask = dt_mask[surface_pred_mask]

        mean_surface_distance = (
            np.mean(distances_mask_to_pred) + np.mean(distances_pred_to_mask)
        ) / 2.0

        bbox_diagonal = np.sqrt(
            np.sum((np.array(mask.shape) * np.array(self.spacing)) ** 2)
        )
        
        return mean_surface_distance / (bbox_diagonal + EPSILON)

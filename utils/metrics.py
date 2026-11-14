import logging

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from scipy.spatial.distance import cdist, directed_hausdorff

# import cv2

# def extract_boundary_with_erosion(mask, dilation_radius=1):
#     """
#     Extract the boundary pixels from a binary mask using morphological operations.

#     Args:
#     - mask (numpy array): Binary mask (2D array).
#     - dilation_radius (int): Radius for dilation to find boundaries.

#     Returns:
#     - numpy array: Binary mask representing the boundary pixels.
#     """
#     kernel = np.ones((3, 3), np.uint8)  # Kernel for dilation and erosion
#     dilated_mask = cv2.dilate(mask, kernel, iterations=dilation_radius)
#     eroded_mask = cv2.erode(mask, kernel, iterations=dilation_radius)

#     # Boundary is the difference between the dilated and eroded versions
#     boundary = dilated_mask - eroded_mask

#     return boundary

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

    # Apply binary dilation to the mask
    structure = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1))
    dilated_mask = binary_dilation(mask, structure=structure)

    # Subtract the original mask from the dilated mask to get the boundary
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


# class LocalRefinementError:
#     def __call__(self, segmentation_a, segmentation_b, pixel):
#         region_a = segmentation_a == segmentation_a[pixel]
#         region_b = segmentation_b == segmentation_b[pixel]
#         error = np.sum(region_a != region_b) / np.sum(region_a | region_b)
#         return error


# class GlobalConsistencyError:
#     def __call__(self, predicted_mask, ground_truth_mask):
#         if predicted_mask.shape != ground_truth_mask.shape:
#             raise ValueError(
#                 "Predicted mask and ground truth mask must have the same dimensions."
#             )

#         gce = 0.0
#         total_pixels = predicted_mask.size
#         local_refinement_error = LocalRefinementError()

#         for pixel in np.ndindex(predicted_mask.shape):
#             error_1 = local_refinement_error(predicted_mask, ground_truth_mask, pixel)
#             error_2 = local_refinement_error(ground_truth_mask, predicted_mask, pixel)
#             gce += min(error_1, error_2)

#         gce /= total_pixels
#         return gce


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


# class MeanAverageSurfaceDistance:
#     """
#     A class to compute the Mean Average Surface Distance (MASD) between predicted and ground truth masks.

#     Attributes:
#     -----------
#     dilation_radius : int
#         The radius used for dilating the mask boundaries.
#     threshold : int, optional
#         The threshold value to binarize the masks. Default is 0.

#     Methods:
#     --------
#     __call__(predicted_mask, ground_truth_mask):
#         Computes the Mean Average Surface Distance between the predicted and ground truth masks.

#     Parameters:
#     -----------
#     predicted_mask : np.ndarray
#         The predicted binary mask.
#     ground_truth_mask : np.ndarray
#         The ground truth binary mask.

#     Returns:
#     --------
#     float
#         The Mean Average Surface Distance between the predicted and ground truth masks.
#     """

#     def __init__(self, dilation_radius, threshold=0):
#         self.dilation_radius = dilation_radius
#         self.threshold = threshold

#     def __call__(self, predicted_mask, ground_truth_mask):
#         """
#         Args:
#             predicted_mask (np.ndarray): The predicted binary mask.
#             ground_truth_mask (np.ndarray): The ground truth binary mask.

#         Returns:
#             float: The average surface distance between the boundaries of the predicted and ground truth masks.
#                    Returns 0.0 if both masks have no boundary pixels.
#                    Returns np.inf if one mask has boundary pixels and the other does not.

#         Raises:
#             ValueError: If the predicted mask and ground truth mask do not have the same dimensions.
#         """
#         if predicted_mask.shape != ground_truth_mask.shape:
#             raise ValueError(
#                 "Predicted mask and ground truth mask must have the same dimensions."
#             )

#         if predicted_mask.dtype != np.uint8:
#             predicted_mask = (predicted_mask >= self.threshold).astype(np.uint8)
#         if ground_truth_mask.dtype != np.uint8:
#             ground_truth_mask = (ground_truth_mask >= self.threshold).astype(np.uint8)

#         pred_boundary = extract_boundary(predicted_mask, self.dilation_radius)
#         gt_boundary = extract_boundary(ground_truth_mask, self.dilation_radius)

#         pred_boundary_coords = np.column_stack(np.where(pred_boundary > 0))
#         gt_boundary_coords = np.column_stack(np.where(gt_boundary > 0))

#         if len(pred_boundary_coords) == 0 and len(gt_boundary_coords) == 0:
#             return 0.0
#         elif len(pred_boundary_coords) == 0 or len(gt_boundary_coords) == 0:
#             return np.inf

#         distances_pred_to_gt = cdist(pred_boundary_coords, gt_boundary_coords).min(
#             axis=1
#         )
#         distances_gt_to_pred = cdist(gt_boundary_coords, pred_boundary_coords).min(
#             axis=1
#         )

#         mean_asd = (np.mean(distances_pred_to_gt) + np.mean(distances_gt_to_pred)) / 2

#         return mean_asd


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

        # Binarize the masks
        if pred_mask.dtype != np.uint8:
            pred_mask = (pred_mask >= self.threshold).astype(np.uint8)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        # Calculate the distance transform for the ground truth and predicted masks
        dt_mask = distance_transform_edt(~mask, sampling=self.spacing)
        dt_pred_mask = distance_transform_edt(~pred_mask, sampling=self.spacing)

        # Extract the surface points of the masks
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

        # Calculate the distances from the surface points of one mask to the other
        distances_mask_to_pred = dt_pred_mask[surface_mask]
        distances_pred_to_mask = dt_mask[surface_pred_mask]

        # Calculate the mean surface distance
        mean_surface_distance = (
            np.mean(distances_mask_to_pred) + np.mean(distances_pred_to_mask)
        ) / 2.0

        # Normalize by the diagonal of the bounding box of the union of the masks
        bbox_diagonal = np.sqrt(
            np.sum((np.array(mask.shape) * np.array(self.spacing)) ** 2)
        )
        
        return mean_surface_distance / (bbox_diagonal + EPSILON)

import numpy as np
import torch
from .utils_masks import calculate_weighted_iou_gpu


def calculate_area(mask):
    """
    Calculate the percentage area of the mask relative to the total image.

    Args:
        mask (torch.Tensor): Binary mask tensor.

    Returns:
        float: Percentage area of the mask.
    """
    mask_area = torch.sum(mask).item()
    total_area = mask.numel()  # Total number of pixels in the mask
    return (mask_area / total_area) * 100


def get_cluster(masks, heatmap, overlap_threshold=0.2):
    """
    Combine masks based on their overlap ratio and return combined masks ordered by their IoU score.

    Args:
        masks (list): List of masks in the format of binary numpy arrays.
        heatmap (torch.Tensor): Heatmap array.
        overlap_threshold (float): Minimum overlap ratio required to combine masks.

    Returns:
        list: List of combined masks ordered by their IoU score (as torch tensors).
        dict: Dictionary of cluster areas.
        dict: Dictionary of cluster scores.
    """
    # Convert masks to torch tensors and move to GPU
    masks_tensor = [torch.tensor(mask['segmentation'], dtype=torch.float32).cuda() for mask in masks]

    # Ensure heatmap is a torch tensor and move to GPU if not already
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.tensor(heatmap, dtype=torch.float32).cuda()

    num_masks = len(masks_tensor)

    combined_masks_list = []
    used_masks = set()
    for i in range(num_masks):
        if i in used_masks:
            continue
        mask_i = masks_tensor[i]
        combined_mask = mask_i.clone()
        used_masks.add(i)
        for j in range(i + 1, num_masks):
            if j in used_masks:
                continue
            mask_j = masks_tensor[j]
            intersection = mask_i * mask_j
            intersection_area = torch.sum(intersection)
            mask_i_area = torch.sum(mask_i)
            mask_j_area = torch.sum(mask_j)
            overlap_ratio_i = intersection_area / mask_i_area if mask_i_area > 0 else 0
            overlap_ratio_j = intersection_area / mask_j_area if mask_j_area > 0 else 0
            if intersection_area > 0 and (overlap_ratio_i > overlap_threshold or overlap_ratio_j > overlap_threshold):
                combined_mask = torch.logical_or(combined_mask, mask_j)
                used_masks.add(j)
        combined_masks_list.append(combined_mask)

    cluster_areas = {idx: calculate_area(combined_mask) for idx, combined_mask in enumerate(combined_masks_list)}
    cluster_scores = {idx: calculate_weighted_iou_gpu(combined_mask, heatmap).cpu().item() for idx, combined_mask in enumerate(combined_masks_list)}

    # Sort clusters by weighted IoU scores
    sorted_cluster_indices = sorted(range(len(combined_masks_list)), key=lambda idx: cluster_scores[idx], reverse=True)

    # Return the clusters as combined masks, ordered by their IoU score
    sorted_combined_masks = [combined_masks_list[idx] for idx in sorted_cluster_indices]

    return sorted_combined_masks, cluster_areas, cluster_scores


def filter_clusters_by_elbow(cluster_scores):
    """
    Filter clusters by identifying the elbow point in their scores.

    Args:
        cluster_scores (dict): Dictionary with cluster indices and their weighted IoU scores.

    Returns:
        list: List of relevant cluster indices.
    """
    sorted_scores = sorted(cluster_scores.items(), key=lambda item: item[1], reverse=True)
    scores = [score for _, score in sorted_scores]

    # Find the elbow point
    n_points = len(scores)
    all_coords = np.vstack((range(n_points), scores)).T
    first_point = all_coords[0]
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))
    vec_from_first = all_coords - first_point
    scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    elbow_index = np.argmax(dist_to_line)

    relevant_clusters = [sorted_scores[i][0] for i in range(elbow_index)]
    return relevant_clusters


def return_combined_clusters(relevant_clusters, sorted_combined_masks, image):
    """
    Display the combined clusters on the background image.

    Args:
        relevant_clusters (list): List of indices of relevant clusters.
        sorted_combined_masks (list): List of combined masks.
        image (np.ndarray): The original RGB image in the range [0, 255].
    """
    combined_mask = np.zeros(image.shape[:2], dtype=bool)

    for cluster_idx in relevant_clusters:
        mask = sorted_combined_masks[cluster_idx].cpu().numpy().astype(bool)  # Move the mask to CPU and convert to bool
        combined_mask = np.logical_or(combined_mask, mask)

    # Create an RGBA image for display
    rgba_image = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
    rgba_image[..., :3] = image
    rgba_image[..., 3] = 255  # Set alpha channel to 255 for the background image

    # Create an overlay with the mask highlighted in red
    overlay = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
    overlay[..., 0] = combined_mask * 255  # Red channel
    overlay[..., 3] = combined_mask * 128  # Alpha channel (50% transparency)

    # Combine the original image with the overlay
    combined_image = rgba_image.copy()
    alpha_factor = overlay[..., 3] / 255.0
    for c in range(3):  # Apply for R, G, B channels
        combined_image[..., c] = (1 - alpha_factor) * rgba_image[..., c] + alpha_factor * overlay[..., c]

    # Clip the values to ensure they are in the valid range [0, 255]
    combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)

    return combined_image
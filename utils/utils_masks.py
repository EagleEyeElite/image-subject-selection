import torch
import numpy as np
import torch.nn.functional as F

def calculate_weighted_iou_gpu(mask, heatmap, exponent=2):
    """
    Calculate the weighted IoU using GPU acceleration with PyTorch.

    Args:
        mask (torch.Tensor): Binary mask tensor.
        heatmap (torch.Tensor): Heatmap tensor.
        exponent (int): Exponent to apply to the heatmap values.

    Returns:
        torch.Tensor: Weighted IoU score (still on GPU).
    """
    # Ensure both mask and heatmap have the same dimensions
    if mask.shape != heatmap.shape:
        heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=mask.shape[-2:], mode='bilinear', align_corners=False).squeeze()

    # Ensure both tensors are on the same device
    mask = mask.to(heatmap.device)

    #heatmap_transformed = torch.pow(heatmap, exponent)
    intersection = mask * heatmap
    union = mask + heatmap - intersection

    intersection_sum = torch.sum(intersection)
    mask_sum = torch.sum(mask)
    union_sum = torch.sum(union)

    if union_sum == 0:
        return torch.tensor(0.0).cuda()

    iou = intersection_sum / union_sum
    weight = intersection_sum / mask_sum

    weighted_iou = iou * weight
    return weighted_iou


def evaluate_heatmap_alignment(masks, heatmap):
    """
    Evaluate the alignment of heatmap with masks using GPU acceleration.

    Args:
        masks (list): List of annotations with binary mask tensors.
        heatmap (np.ndarray): Heatmap array.

    Returns:
        list: Sorted IoU scores.
        list: Sorted masks.
    """
    iou_scores = []

    # Convert heatmap to torch tensor and move to GPU
    heatmap_tensor = torch.tensor(heatmap, dtype=torch.float32).cuda()

    for idx, ann in enumerate(masks):
        mask = torch.tensor(ann['segmentation'], dtype=torch.float32).cuda()
        weighted_iou = calculate_weighted_iou_gpu(mask, heatmap_tensor)
        iou_scores.append((idx, weighted_iou))

    # Sort masks by IoU score
    iou_scores.sort(key=lambda x: x[1], reverse=True)
    sorted_indices = [idx for idx, _ in iou_scores]
    sorted_masks = [masks[idx] for idx in sorted_indices]
    sorted_iou_scores = [iou for _, iou in iou_scores]

    return sorted_iou_scores, sorted_masks

def return_selected_masks(relevant_clusters, sorted_combined_masks):
    """
    Return the combined mask for the relevant clusters.

    Args:
        relevant_clusters (list): List of indices of relevant clusters.
        sorted_combined_masks (list): List of combined masks.

    Returns:
        np.ndarray: Combined binary mask.
    """
    combined_mask = np.zeros(sorted_combined_masks[0].shape, dtype=bool)

    for cluster_idx in relevant_clusters:
        mask = sorted_combined_masks[cluster_idx].cpu().numpy().astype(bool)  # Move the mask to CPU and convert to bool
        combined_mask = np.logical_or(combined_mask, mask)

    return combined_mask.astype(np.uint8) * 255  # Convert boolean mask to uint8 format with values 0 or 255
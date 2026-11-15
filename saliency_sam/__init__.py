"""Automatic image segmentation: SAM proposes every segment, a saliency map picks.

SAM's automatic mask generator segments the whole image without prompts. A
classifier's saliency map (LayerCAM, GradCAM, LRP) then scores each mask by
how much of the relevance it covers; overlapping masks are clustered and the
clusters before the elbow of the score curve form the segmentation.
"""

from .models import choose_device, load_image, load_resnet, load_sam, predict, to_tensor
from .segmentation import Cluster, SegmentationResult, mask_generator, segment

__all__ = [
    "Cluster",
    "SegmentationResult",
    "choose_device",
    "load_image",
    "load_resnet",
    "load_sam",
    "mask_generator",
    "predict",
    "segment",
    "to_tensor",
]

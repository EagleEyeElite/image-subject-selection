from .segment_eval_handler import SegmentEvalHandler
import numpy as np
from PIL import Image


class IOU(SegmentEvalHandler):
    def __call__(self, prompt):
        # calculate intersection over union
        prompt.segmentation_evaluation = []
        ground_truth = prompt.mask.copy()
        ground_truth[ground_truth > 0] = 1

        for i in prompt.segmentation_masks:

            intersection = np.logical_and(ground_truth, i)
            union = np.logical_or(ground_truth, i)
            iou = np.sum(intersection) / np.sum(union)
            prompt.segmentation_evaluation.append(iou)

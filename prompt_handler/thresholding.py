from .point_generation_handler import PointGenerationHandler
import numpy as np
from skimage.transform import resize


class ThresholdSaliencyPoints(PointGenerationHandler):
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def __call__(self, prompt):
        upsacale_size = (
            prompt.original_image.shape[0], prompt.original_image.shape[1])
        # Bilinear interpolation
        heatmap = resize(prompt.heatmap, upsacale_size, order=1)

        # Select points above the threshold
        points = np.argwhere(heatmap > self.threshold)

        prompt.points = points
        prompt.point_labels = np.ones(len(points))
        return prompt

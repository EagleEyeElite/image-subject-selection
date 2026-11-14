from .point_generation_handler import PointGenerationHandler
import numpy as np
from skimage.transform import resize


class NegativeGridThresholdSaliencyPoints(PointGenerationHandler):
    def __init__(self, grid_size=4, grid_threshold=0.4, saliency_threshold=0.9):
        self.grid_size = grid_size
        self.grid_threshold = grid_threshold
        self.saliency_threshold = saliency_threshold

    def __call__(self, prompt):
        upscale_size = (
            prompt.original_image.shape[0], prompt.original_image.shape[1])
        # Bilinear interpolation
        heatmap = resize(prompt.heatmap, upscale_size, order=1)

        points = []
        point_labels = []

        # NegativeGrid method
        x_spacing = upscale_size[1] // self.grid_size
        y_spacing = upscale_size[0] // self.grid_size

        # Iterate over the grid and place points
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = i * x_spacing
                y = j * y_spacing

                # Check if the saliency map value is below the threshold
                if heatmap[y, x] < self.grid_threshold:
                    points.append((y, x))
                    point_labels.append(0)

        # ThresholdSaliencyPoints method
        # Select points above the saliency threshold
        saliency_points = np.argwhere(heatmap > self.saliency_threshold)

        # Add saliency points to the list with label 1
        for point in saliency_points:
            points.append((point[0], point[1]))
            point_labels.append(1)

        prompt.points = np.array(points)
        prompt.point_labels = np.array(point_labels)
        return prompt

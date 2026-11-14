from .point_generation_handler import PointGenerationHandler
import numpy as np
from skimage.transform import resize


class NegativeGridMaxValue(PointGenerationHandler):
    def __init__(self, grid_size=4, threshold=0.4):
        self.grid_size = grid_size
        self.threshold = threshold

    def __call__(self, prompt):
        upsacale_size = (
            prompt.original_image.shape[0], prompt.original_image.shape[1])
        # bilinear interpolation
        heatmap = resize(prompt.heatmap, upsacale_size, order=1)

        # MaxValue method
        hotspot = np.unravel_index(
            np.argmax(heatmap, axis=None), heatmap.shape)
        points = [hotspot]
        point_labels = [1]

        # NegativeGrid method
        x_spacing = upsacale_size[1] // self.grid_size
        y_spacing = upsacale_size[0] // self.grid_size

        # Iterate over the grid and place points
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = i * x_spacing
                y = j * y_spacing

                # Check if the saliency map value is below the threshold
                if heatmap[y, x] < self.threshold:
                    points.append((y, x))
                    point_labels.append(0)

        prompt.points = np.array(points)
        prompt.point_labels = np.array(point_labels)
        return prompt

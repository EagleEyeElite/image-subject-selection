from .point_generation_handler import PointGenerationHandler
import numpy as np
from skimage.transform import resize
from skimage import measure


class NegativeGridContourDetection(PointGenerationHandler):
    def __init__(self, grid_size=4, threshold=0.4, contour_threshold=0.5):
        self.grid_size = grid_size
        self.threshold = threshold
        self.contour_threshold = contour_threshold

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
                if heatmap[y, x] < self.threshold:
                    points.append((y, x))
                    point_labels.append(0)

        # ContourDetection method
        # Apply threshold to create binary mask
        binary_map = heatmap > self.contour_threshold

        # Find contours
        contours = measure.find_contours(binary_map, level=0.5)

        bounding_boxes = []
        if len(contours) > 0:
            for contour in contours:
                y, x = contour[:, 0], contour[:, 1]
                x_min, x_max = np.min(x), np.max(x)
                y_min, y_max = np.min(y), np.max(y)
                bounding_box = {
                    'x': int(x_min),
                    'y': int(y_min),
                    'width': int(x_max - x_min),
                    'height': int(y_max - y_min),
                    'label': ''
                }
                bounding_boxes.append(bounding_box)

        prompt.points = np.array(points)
        prompt.point_labels = np.array(point_labels)
        prompt.bounding_boxes = bounding_boxes
        return prompt

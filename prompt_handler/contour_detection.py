from .point_generation_handler import PointGenerationHandler
import numpy as np
from skimage.transform import resize
from skimage import measure


class ContourDetection(PointGenerationHandler):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, prompt):
        upscale_size = (
            prompt.original_image.shape[0], prompt.original_image.shape[1])
        # Bilinear interpolation
        heatmap = resize(prompt.heatmap, upscale_size, order=1)

        # Apply threshold to create binary mask
        binary_map = heatmap > self.threshold

        # Find contours
        contours = measure.find_contours(binary_map, level=0.5)

        if len(contours) == 0:
            bounding_boxes = []
        else:
            bounding_boxes = []
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

        prompt.bounding_boxes = bounding_boxes
        return prompt

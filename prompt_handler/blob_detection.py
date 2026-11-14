from .point_generation_handler import PointGenerationHandler
import numpy as np
from skimage.feature import blob_log
from skimage.color import rgb2gray
from skimage.transform import resize


class BlobDetection(PointGenerationHandler):
    def __init__(self, min_sigma=1, max_sigma=2, num_sigma=2, threshold=0.3, overlap=0.5):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = threshold
        self.overlap = overlap
        self.heatmap_threshold = heatmap_threshold

    def __call__(self, prompt):
        upscale_size = (
            prompt.original_image.shape[0], prompt.original_image.shape[1])
        heatmap = resize(prompt.heatmap, upscale_size, order=1)

        if len(prompt.original_image.shape) == 3:
            gray_image = rgb2gray(prompt.original_image)
        else:
            gray_image = prompt.original_image

        # Detect blobs in the heatmap
        blobs = blob_log(gray_image, min_sigma=self.min_sigma, max_sigma=self.max_sigma,
                         num_sigma=self.num_sigma, threshold=self.threshold, overlap=self.overlap)

        # Filter blobs based on heatmap values
        bounding_boxes = []
        points = []
        for blob in blobs:
            y, x, r = blob
            if heatmap[int(y), int(x)] > self.heatmap_threshold:
                x_min = max(int(x - r), 0)
                y_min = max(int(y - r), 0)
                x_max = min(int(x + r), gray_image.shape[1])
                y_max = min(int(y + r), gray_image.shape[0])
                width = x_max - x_min
                height = y_max - y_min
                bounding_boxes.append(
                    {'x': x_min, 'y': y_min, 'width': width, 'height': height, 'label': ''})
                points.append([y, x])

        prompt.bounding_boxes = bounding_boxes
        prompt.points = np.array(points)
        prompt.point_labels = np.ones(len(points))
        return prompt

# Usage:
# Adjust the following parameters to optimize detection
# min_sigma: Increase to ignore smaller features
# max_sigma: Decrease to ignore larger features
# num_sigma: Decrease to reduce the number of sigma values tested
# threshold: Increase to make blob detection more stringent
# overlap: Decrease to reduce overlapping blobs
# heatmap_threshold: Increase to ignore lower saliency regions

# Example:
# blob_detection = BlobDetection(min_sigma=2, max_sigma=30, num_sigma=5, threshold=0.2, overlap=0.3, heatmap_threshold=0.6)

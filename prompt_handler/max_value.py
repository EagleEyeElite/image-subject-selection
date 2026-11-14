from .point_generation_handler import PointGenerationHandler
import numpy as np
from skimage.transform import resize
from scipy.ndimage import zoom
from PIL import Image


class MaxValue(PointGenerationHandler):
    def __call__(self, prompt):

        required_shape = prompt.original_image.shape[:2]
        heatmap = prompt.heatmap.copy()
        # zoom heatmap to original image size
        zoom_factors = (
            required_shape[0] / heatmap.shape[0],
            required_shape[1] / heatmap.shape[1]
        )
        heatmap = zoom(heatmap, zoom_factors, order=0)

        flat_heatmap = heatmap.flatten()
        max_value_index = np.argmax(flat_heatmap)
        restored_index = np.unravel_index(
            max_value_index, heatmap.shape)
        restored_index = (restored_index[1], restored_index[0])
        prompt.points = np.array([restored_index])
        prompt.point_labels = np.array([1])
        return prompt

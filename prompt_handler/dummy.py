from .point_generation_handler import PointGenerationHandler
import numpy as np
from skimage.transform import resize
from scipy.ndimage import zoom
from PIL import Image


class Dummy(PointGenerationHandler):
    def __call__(self, prompt):

        required_shape = prompt.original_image.shape
        hotspot = np.array([required_shape[0] // 2, required_shape[1] // 2])

        prompt.points = np.array([hotspot])
        prompt.point_labels = np.array([1])

        return prompt

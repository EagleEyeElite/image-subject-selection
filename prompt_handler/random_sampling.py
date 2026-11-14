import numpy as np
from skimage.transform import resize
from .point_generation_handler import PointGenerationHandler


class RandomSamplingWithImportance(PointGenerationHandler):
    def __init__(self, num_points=2):
        self.num_points = num_points

    def __call__(self, prompt):
        upscale_size = (
            prompt.original_image.shape[0], prompt.original_image.shape[1])
        # Bilinear interpolation
        heatmap = resize(prompt.heatmap, upscale_size, order=1)

        # Flatten the heatmap and get the probabilities
        flat_heatmap = heatmap.flatten()
        probabilities = flat_heatmap / np.sum(flat_heatmap)

        # Ensure probabilities are non-negative and sum up to 1
        probabilities /= np.sum(probabilities)

        # Randomly sample indices based on the probabilities
        sampled_indices = np.random.choice(
            np.arange(len(flat_heatmap)),
            size=self.num_points,
            replace=False,
            p=probabilities
        )

        # Convert flat indices back to 2D coordinates
        points = np.array(np.unravel_index(sampled_indices, heatmap.shape)).T

        prompt.points = points
        prompt.point_labels = np.ones(len(points))
        return prompt

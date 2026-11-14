import numpy as np
from skimage.transform import resize
from .point_generation_handler import PointGenerationHandler


class NegativeGridRandomSamplingWithImportance(PointGenerationHandler):
    def __init__(self, grid_size=4, threshold=0.4, num_points=2):
        self.grid_size = grid_size
        self.threshold = threshold
        self.num_points = num_points

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

        # RandomSamplingWithImportance method
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
        sampled_points = np.array(np.unravel_index(
            sampled_indices, heatmap.shape)).T

        # Add sampled points to the list with label 1
        for point in sampled_points:
            points.append((point[0], point[1]))
            point_labels.append(1)

        prompt.points = np.array(points)
        prompt.point_labels = np.array(point_labels)
        return prompt

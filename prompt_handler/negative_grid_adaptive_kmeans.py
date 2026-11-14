from .point_generation_handler import PointGenerationHandler
import numpy as np
from skimage.transform import resize
from sklearn.cluster import KMeans


class NegativeGridKMeans(PointGenerationHandler):
    def __init__(self, grid_size=4, threshold=0.4, num_clusters=1):
        self.grid_size = grid_size
        self.threshold = threshold
        self.num_clusters = num_clusters

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

        # Flatten the heatmap and get coordinates
        flat_heatmap = heatmap.flatten()

        # Ensure all values are non-negative
        flat_heatmap = np.clip(flat_heatmap, a_min=0, a_max=None)

        # Normalize the heatmap to sum to 1 (if not already zero)
        if np.sum(flat_heatmap) > 0:
            flat_heatmap = flat_heatmap / np.sum(flat_heatmap)

        coords = np.array(np.unravel_index(
            np.arange(flat_heatmap.size), heatmap.shape
        )).T

        # Apply KMeans with the specified number of clusters
        num_clusters = min(self.num_clusters, len(flat_heatmap))
        if num_clusters > 0:
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(
                coords, sample_weight=flat_heatmap
            )
            hotspots = kmeans.cluster_centers_
            for hotspot in hotspots:
                points.append((int(hotspot[1]), int(hotspot[0])))
                point_labels.append(1)

        prompt.points = np.array(points)
        prompt.point_labels = np.array(point_labels)
        return prompt

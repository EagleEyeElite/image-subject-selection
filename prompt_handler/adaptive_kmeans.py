from .point_generation_handler import PointGenerationHandler
import numpy as np
from skimage.transform import resize
from sklearn.cluster import KMeans


class AdaptiveKMeans(PointGenerationHandler):
    def __init__(self, num_clusters=1, threshold=0.5):
        self.num_clusters = num_clusters
        self.threshold = threshold

    def __call__(self, prompt):
        upscale_size = (
            prompt.original_image.shape[0], prompt.original_image.shape[1]
        )
        # Bilinear interpolation
        heatmap = resize(prompt.heatmap, upscale_size, order=1)

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
        # Ensure we do not ask for more clusters than available points
        num_clusters = min(self.num_clusters, len(flat_heatmap))
        if num_clusters > 0:
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(
                coords, sample_weight=flat_heatmap
            )
            hotspots = kmeans.cluster_centers_
        else:
            hotspots = np.array([])

        prompt.points = hotspots
        prompt.point_labels = np.ones(len(hotspots))
        return prompt

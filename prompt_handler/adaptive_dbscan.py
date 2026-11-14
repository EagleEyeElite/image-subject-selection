from .point_generation_handler import PointGenerationHandler
import numpy as np
from collections import Counter
from skimage.transform import resize
from sklearn.cluster import DBSCAN


class AdaptiveDBSCAN(PointGenerationHandler):
    def __init__(self, eps=0.5, min_samples=5, saliency_threshold=0.5, max_clusters=5):
        self.eps = eps
        self.min_samples = min_samples
        self.saliency_threshold = saliency_threshold
        self.max_clusters = max_clusters

    def __call__(self, prompt):
        upsacale_size = (
            prompt.original_image.shape[0], prompt.original_image.shape[1])
        # Bilinear interpolation
        heatmap = resize(prompt.heatmap, upsacale_size, order=1)

        # Apply threshold to filter low saliency points
        high_saliency_mask = heatmap >= self.saliency_threshold

        # Flatten the heatmap and create coordinates for high saliency points
        y_coords, x_coords = np.indices(heatmap.shape)
        coordinates = np.column_stack((y_coords.ravel(), x_coords.ravel()))
        high_saliency_coordinates = coordinates[high_saliency_mask.ravel()]
        high_saliency_values = heatmap[high_saliency_mask]

        if len(high_saliency_coordinates) == 0:
            prompt.points = np.array([])
            prompt.point_labels = np.array([])
            return prompt

        # Normalize the coordinates for DBSCAN
        normalized_coordinates = high_saliency_coordinates / \
            np.max(coordinates, axis=0)

        # Apply DBSCAN clustering
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(
            normalized_coordinates, sample_weight=high_saliency_values)

        # Get labels and select clusters
        labels = db.labels_
        label_counts = Counter(labels)
        # Exclude noise label (-1)
        if -1 in label_counts:
            del label_counts[-1]

        # Get the largest clusters
        largest_clusters = label_counts.most_common(self.max_clusters)
        selected_labels = [label for label, _ in largest_clusters]

        # Select points from the largest clusters
        selected_points_mask = np.isin(labels, selected_labels)
        selected_points = high_saliency_coordinates[selected_points_mask]

        prompt.points = selected_points
        prompt.point_labels = np.ones(len(selected_points))
        return prompt

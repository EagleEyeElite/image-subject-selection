import numpy as np
from scipy.ndimage import gaussian_filter
from .point_generation_handler import PointGenerationHandler


class PeakPointDetectionHandler(PointGenerationHandler):
    """
    Generates points by detecting peaks in a heatmap.

    Attributes:
        threshold (float): The threshold for detecting peak points.
        sigma (float): The standard deviation for Gaussian smoothing.

    Methods:
        __call__(heatmap, original_image) -> dict:
            Generates points by detecting peaks in a heatmap.

            Args:
                heatmap (np.array): The input heatmap.
                original_image (np.array): The original image corresponding to the heatmap.

            Returns:
                dict: A dictionary containing peak points and their labels.
                    Example:
                    {'points': np.array([[x1, y1], [x2, y2]]),
                     'point_labels': np.array([1, 1])}
    """

    def __init__(self, threshold=0.5, sigma=2):
        self.threshold = threshold
        self.sigma = sigma

    def __call__(self, heatmap, original_image) -> dict:
        # Smooth the heatmap to reduce noise
        smoothed_heatmap = gaussian_filter(heatmap, sigma=self.sigma)

        # Find peak points
        peak_points = np.argwhere(
            smoothed_heatmap > self.threshold * np.max(smoothed_heatmap))

        # Create point_labels (1 for peak points)
        point_labels = np.ones(len(peak_points), dtype=int)

        return {'points': peak_points, 'point_labels': point_labels}


# Test
# if __name__ == "__main__":
#     # Example heatmap
#     heatmap = np.random.rand(10, 10)
#     original_image = np.random.rand(224, 224, 3)
#
#     # Peak point detection
#     peak_handler = PeakPointDetectionHandler(threshold=0.8)
#     peak_points = peak_handler(heatmap, original_image)
#     print("Peak Points:", peak_points)

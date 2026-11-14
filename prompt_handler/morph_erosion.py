from .point_generation_handler import PointGenerationHandler
import numpy as np
from skimage.morphology import erosion, square
from skimage.measure import regionprops, label


class MorphologicalErosion(PointGenerationHandler):
    def __init__(self, erosion_size=3):
        self.erosion_size = erosion_size

    def __call__(self, prompt):
        # Erode the saliency map
        eroded_heatmap = erosion(prompt.heatmap, square(self.erosion_size))

        # Label connected regions
        labeled_heatmap = label(eroded_heatmap)

        # Find the center of each non-zero block
        regions = regionprops(labeled_heatmap)
        hotspots = np.array([region.centroid for region in regions])

        prompt.points = hotspots
        prompt.point_labels = np.ones(len(hotspots))
        return prompt

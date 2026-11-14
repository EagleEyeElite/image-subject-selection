from utils.prompt import Prompt
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import zoom

TITLE_FONT_SIZE = 10


class Visualizer:
    def __init__(self, data: list[Prompt]):
        self.data = data
        self.run()

    def run(self):
        self.create_plot()
        self.place_original_images()
        self.show_points()
        self.place_heatmaps()
        self.place_masks()
        self.place_segmentations()

    def create_plot(self):
        self.fig, self.axs = plt.subplots(
            nrows=len(self.data),
            ncols=2 + self.data[0].segmentation_count,
            figsize=(10, 10),
        )

    def place_original_images(self):
        for i, prompt in enumerate(self.data):
            self.axs[i, 0].imshow(prompt.original_image)
            self.axs[i, 0].set_title(
                f"Original: {prompt.original_image_class}", fontsize=TITLE_FONT_SIZE)
            self.axs[i, 0].axis("off")

    def place_heatmaps(self):
        for i, prompt in enumerate(self.data):
            required_shape = prompt.original_image.shape
            zoom_factors = (
                required_shape[0] / prompt.heatmap.shape[0],
                required_shape[1] / prompt.heatmap.shape[1]
            )
            print(zoom_factors)
            print(prompt.heatmap.shape)
            print(required_shape)
            resized_heatmap = zoom(prompt.heatmap, zoom_factors, order=0)
            self.axs[i, 1].imshow(
                resized_heatmap, cmap='viridis')
            first_label = prompt.predicted_class_label.split(",")[0]
            self.axs[i, 1].set_title(
                f"Heatmap: {first_label}", fontsize=TITLE_FONT_SIZE)

    def show_points(self, marker_size=375):
        for i, prompt in enumerate(self.data):
            pos_points = prompt.points[prompt.point_labels == 1]
            neg_points = prompt.points[prompt.point_labels == 0]
            self.axs[i, 0].scatter(pos_points[:, 0], pos_points[:, 1], color='green',
                                   marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
            self.axs[i, 0].scatter(neg_points[:, 0], neg_points[:, 1], color='red',
                                   marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    def place_masks(self):
        for i, prompt in enumerate(self.data):
            mask = prompt.mask
            mask = prompt.mask.copy()
            mask = np.flip(mask, axis=0)
            self.axs[i, 1].contour(
                mask[::-1], levels=[0.5], colors='w', alpha=0.6)
            self.axs[i, 1].axis("off")

    def place_segmentations(self):
        for i, prompt in enumerate(self.data):
            for j in range(prompt.segmentation_count):
                mask = prompt.segmentation_masks[j]
                color = np.array([30/255, 144/255, 255/255, 0.6])
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                self.axs[i, j + 2].imshow(mask_image)
                self.axs[i, j + 2].set_title(
                    f"Score: {prompt.segmentation_evaluation[j]:.2f}", fontsize=TITLE_FONT_SIZE)
                self.axs[i, j + 2].axis("off")

    # def show_box(box, ax):
    #     x0, y0 = box[0], box[1]
    #     w, h = box[2] - box[0], box[3] - box[1]
    #     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

    def __call__(self):
        return self.fig

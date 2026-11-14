import torch
import torch.nn.functional as F
from .heatmap_handler import HeatmapHandler

# TODO fix methodology


class GradCAMHeatmap(HeatmapHandler):
    def normalize(self, masks):
        """
        Min-max normalization for each channel
        Args:
            masks (torch.Tensor): Masks BxCxWxH.

        """
        mins = masks.amin(dim=(-2, -1))
        maxs = masks.amax(dim=(-2, -1))
        # expand dims to [N activations, 1, 1]
        maxs, mins = maxs[:, None, None], mins[:, None, None]

        return (masks - mins) / (maxs - mins + 1e-8)

    def __call__(self, img_tensor):
        size = img_tensor.shape[-2:]
        with torch.no_grad():
            pred = self.model(img_tensor)
            pred = torch.argmax(pred, dim=1).item()
            pred_label = self.imagenet_labels[pred]
            pred_label = pred_label.split(",")[0]

        heatmap = self.layer_gc.attribute(
            img_tensor, int(pred), relu_attributions=True
        ).detach()
        heatmap = self.normalize(heatmap)
        if self.threshold is not None:
            heatmap[heatmap < self.threshold] = 0
        if self.interpolation is not None:
            heatmap = F.interpolate(
                heatmap, size=size, mode=self.interpolation)
        return heatmap.squeeze(0), pred_label

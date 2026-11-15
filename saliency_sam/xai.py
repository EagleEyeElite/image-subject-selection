"""Saliency maps for the predicted class.

Each method is a callable ``method(img_tensor, class_idx) -> torch.Tensor``
returning a 2D map at its own resolution; ``to_image_size`` turns it into a
[0, 1] map the size of the image.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchcam.methods
from captum.attr import LayerGradCam


class LayerCAM:
    """LayerCAM (torchcam) on one layer, e.g. ``resnet.layer4``."""

    def __init__(self, model, layer):
        self.model = model
        self.layer = layer

    def __call__(self, img_tensor, class_idx):
        with torchcam.methods.LayerCAM(self.model, target_layer=self.layer) as cam:
            output = self.model(img_tensor)
            return cam(class_idx, output)[0][0].detach()


class GradCAM:
    """GradCAM (captum) on one layer, e.g. ``resnet.layer4``."""

    def __init__(self, model, layer):
        self.gradcam = LayerGradCam(model, layer)

    def __call__(self, img_tensor, class_idx):
        heatmap = self.gradcam.attribute(img_tensor, int(class_idx), relu_attributions=True)
        return heatmap.detach().squeeze(0).squeeze(0)


class LRP:
    """Layer-wise relevance propagation (zennit).

    Without ``layer`` the relevance at the input is returned, otherwise the
    relevance at that layer (e.g. ``"layer2.0.conv2"``), averaged over channels.
    """

    def __init__(self, model, composite, layer: str | None = None):
        self.model = model
        self.composite = composite
        self.layer = layer

    def __call__(self, img_tensor, class_idx):
        img_tensor = img_tensor.detach().requires_grad_(True)
        with self.composite.context(self.model) as modified_model:
            if self.layer is None:
                modified_model(img_tensor)[0, class_idx].backward()
                return img_tensor.grad.squeeze(0).mean(dim=0).detach()

            module = dict(modified_model.named_modules())[self.layer]
            captured = {}

            def keep_output(_module, _input, output):
                output = output[-1] if isinstance(output, tuple) else output
                output.retain_grad()
                captured["output"] = output

            handle = module.register_forward_hook(keep_output)
            try:
                modified_model(img_tensor)[0, class_idx].backward()
            finally:
                handle.remove()
            return captured["output"].grad.squeeze(0).mean(dim=0).detach()


def to_image_size(heatmap: torch.Tensor, shape: tuple[int, ...]) -> np.ndarray:
    """Clip negative relevance, scale to [0, 1], resize bilinearly to ``shape[:2]``."""
    heatmap = torch.relu(heatmap.float())
    heatmap = heatmap / heatmap.max().clamp_min(1e-12)
    heatmap = F.interpolate(heatmap[None, None], size=tuple(shape[:2]), mode="bilinear")
    return heatmap[0, 0].cpu().numpy()

"""Loading SAM, the ResNet50 classifier and images."""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from segment_anything import sam_model_registry
from torchvision.models import ResNet50_Weights

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/saliency_sam/models.py

# scripts/download_checkpoints.sh puts the weights here; the Docker image sets
# CHECKPOINT_DIR=/usr/models.
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"))

SAM_CHECKPOINTS = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}
RESNET_CHECKPOINT = "resnet50-0676ba61.pth"  # = ResNet50_Weights.IMAGENET1K_V1

IMAGENET_LABELS: list[str] = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

# ImageNet normalization; no resizing, the saliency map is scaled to the image.
to_tensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_sam(model_type: str = "vit_b", checkpoint: str | Path | None = None, device=None):
    """SAM model of the given type; the checkpoint defaults to CHECKPOINT_DIR."""
    checkpoint = checkpoint or CHECKPOINT_DIR / SAM_CHECKPOINTS[model_type]
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"{checkpoint} not found, run scripts/download_checkpoints.sh")
    return sam_model_registry[model_type](checkpoint=str(checkpoint)).to(device or choose_device())


def load_resnet(checkpoint: str | Path | None = None, device=None) -> torch.nn.Module:
    """ResNet50 with ImageNet weights, in eval mode.

    Uses the local checkpoint if it exists, otherwise torchvision downloads
    the same weights into its cache.
    """
    checkpoint = Path(checkpoint or CHECKPOINT_DIR / RESNET_CHECKPOINT)
    device = device or choose_device()
    if checkpoint.is_file():
        model = torchvision.models.resnet50()
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    else:
        model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    return model.to(device).eval()


def load_image(path: str | Path) -> np.ndarray:
    """RGB image as a uint8 array (H, W, 3)."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def predict(model: torch.nn.Module, img_tensor: torch.Tensor) -> int:
    """Index of the top ImageNet class."""
    with torch.no_grad():
        return model(img_tensor).argmax().item()

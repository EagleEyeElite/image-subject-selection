import os
import torch
import torchvision
import cv2
from segment_anything import sam_model_registry
import json
import numpy as np
import random

device = 'cuda'


def check_cuda_support():
    """
    Checks if CUDA is available and returns the device to use ('cuda' or 'cpu').

    Returns:
        device (str): The device to use for tensor operations ('cuda' if available, otherwise raises RuntimeError).
        gpu_name (str): The name of the GPU if CUDA is available.

    Raises:
        RuntimeError: If CUDA is not available.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"Using GPU: {gpu_name}")
        return gpu_name
    else:
        raise RuntimeError("CUDA is not available. Tensor operations can only run on CPU. Stopping execution.")


# Function to check if running inside Docker
def check_if_running_in_docker():
    return os.getenv('RUNNING_IN_DOCKER', 'false').lower() == 'true'


def load_models(sam_checkpoint_path, resnet_checkpoint_path, class_labels_path, image_path=None):
    """
    Loads the SAM model, ResNet model, ImageNet labels, and image, and generates masks.

    Args:
        sam_checkpoint_path (str): Path to the SAM model checkpoint.
        resnet_checkpoint_path (str): Path to the ResNet model checkpoint.
        class_labels_path (str): Path to the ImageNet class labels.
        image_path (str, optional): Path to an image; if given, the RGB image is returned as 4th value.

    Returns:
        sam (object): Loaded SAM model.
        mask_generator (object): Created Mask Generator.
        resnet (object): Loaded ResNet model.
        imagenet_labels (object): Loaded ImageNet labels.
        masks (object): Generated masks.
        cam_extractor (object): Created CAM extractor.

    Raises:
        AssertionError: If any of the models or data are not loaded correctly.
    """
    # Load models and image
    sam = load_sam_model(sam_checkpoint_path)
    resnet = load_resnet_model(resnet_checkpoint_path)
    imagenet_labels = load_imagenet_labels(class_labels_path)

    # Test the loading
    assert sam is not None, "SAM model not loaded."
    assert resnet is not None, "ResNet model not loaded."
    assert imagenet_labels is not None, "Imagenet labels not loaded."

    if image_path is not None:
        image_rgb = load_image(image_path)
        assert image_rgb is not None, f"Image not loaded: {image_path}"

    print("All models and data loaded correctly.")

    if image_path is not None:
        return sam, resnet, imagenet_labels, image_rgb
    return sam, resnet, imagenet_labels


# Function to load SAM model
def load_sam_model(checkpoint_path):
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path).to(device)
    return sam


# Function to load ResNet model
def load_resnet_model(checkpoint_path):
    resnet = torchvision.models.resnet50()
    resnet.load_state_dict(torch.load(checkpoint_path, map_location=device))
    resnet.to(device).eval()
    return resnet


# Function to load and process image
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def img_to_cv2(image):
    """
    Convert PIL image to OpenCV format.
    
    Args:
    - image: PIL Image.
    
    Returns:
    - image_cv2: OpenCV image (NumPy array).
    """
    return np.array(image)

def get_random_samples(dataset, sample_size):
    """
    Select a random sample of images from the dataset.

    Args:
    - dataset: The dataset from which to sample.
    - sample_size: The number of samples to select.

    Returns:
    - sample_indices: List of indices for the selected samples.
    """
    sample_indices = random.sample(range(len(dataset)), sample_size)
    return sample_indices


def load_imagenet_labels(labels_path):
    with open(labels_path) as f:
        imagenet_labels = list(json.load(f).values())
        return imagenet_labels
    

def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"

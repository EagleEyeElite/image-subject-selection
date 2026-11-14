import torchvision
from . import device
import json
from .model_handler import ModelHandler


class vgg16(ModelHandler):
    def __init__(self):
        super().__init__(device)

    def load_model(self):
        m = torchvision.models.vgg16(weights="VGG16_Weights.IMAGENET1K_V1")
        m.eval().to(device)
        self.model = m
        self.layer = self.model.features
        return m

    def load_labels(self):
        with open("imagenet_class_index.json") as f:
            imagenet_labels = list(json.load(f).values())
        self.imagenet_labels = imagenet_labels
        return imagenet_labels

    def load_input_size(self):
        return 224

import torchvision
from . import device
import json
from .model_handler import ModelHandler


class resnet(ModelHandler):
    def __init__(self, layer=None):
        # Initialize the superclass with the device and layer
        super().__init__(device, layer)

    def load_model(self):
        # Load the pre-trained ResNet50 model
        m = torchvision.models.resnet50(
            weights="ResNet50_Weights.IMAGENET1K_V1")
        m.eval().to(device)  # Set the model to evaluation mode and move it to the specified device
        self.model = m

        # Set the layer attribute to the specified layer or default to layer4 if not specified
        if not hasattr(self, 'layer'):
            self.layer = self.model.layer4
        else:
            match self.layer:
                case 1:
                    self.layer = self.model.layer1
                case 2:
                    self.layer = self.model.layer2
                case 3:
                    self.layer = self.model.layer3
                case _:
                    self.layer = self.model.layer4

        return m

    def load_labels(self):
        # Load the ImageNet class labels from a JSON file
        with open("imagenet_class_index.json") as f:
            imagenet_labels = list(json.load(f).values())
        self.imagenet_labels = imagenet_labels
        return imagenet_labels

    def load_input_size(self):
        # Return the input size for the ResNet model
        return 224

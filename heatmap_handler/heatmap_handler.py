import abc
from model_handler.model_handler import ModelHandler
from utils.prompt import Prompt

# Abstract base class for handling heatmap generation


class HeatmapHandler(metaclass=abc.ABCMeta):
    def __init__(self, model: ModelHandler, interpolation=None, threshold=None):
        self.model = model.model  # Store the model
        self.layer = model.layer  # Store the specific layer of the model for heatmap generation
        self.imagenet_labels = model.imagenet_labels  # Store ImageNet labels
        self.interpolation = interpolation  # Store interpolation method, if any
        self.threshold = threshold  # Store threshold value, if any

    # Call method to execute the heatmap generation

    def __call__(self, prompt: Prompt) -> Prompt:
        if self.execute is None:
            raise NotImplementedError  # Raise an error if the execute method is not implemented
        # Generate the heatmap using the execute method
        prompt = self.execute(prompt)
        return prompt  # Return the updated prompt

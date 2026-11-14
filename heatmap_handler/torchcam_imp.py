from torchcam.methods import CAM, SmoothGradCAMpp, ScoreCAM, SSCAM, GradCAM, GradCAMpp, XGradCAM, LayerCAM

from model_handler.model_handler import ModelHandler
from .heatmap_handler import HeatmapHandler


# Factory function to generate heatmap extraction method
def method_factory(method, model, layer, labels):
    def res(prompt):
        # Use the specified CAM method in context
        with method(model, target_layer=layer) as cam_extractor:
            # Get the image tensor from the prompt
            img_tensor = prompt.original_image_tensor
            img_tensor.requires_grad = True  # Enable gradient computation
            output = model(img_tensor)  # Get the model output
            prediction = output.max(1)[-1].item()  # Get the top prediction
            # Extract the activation map
            activation_map = cam_extractor(prediction, output)
            # Store the heatmap in the prompt
            prompt.heatmap = activation_map[0][0].cpu().numpy()
            prompt.predicted_class_id = prediction  # Store the predicted class ID
            # Store the predicted class label
            prompt.predicted_class_label = labels[prediction]
            return prompt  # Return the updated prompt
    return res


class torchcam_CAM(HeatmapHandler):
    def __init__(self, model: ModelHandler, interpolation=None, threshold=None):
        super().__init__(model, interpolation, threshold)
        self.execute = method_factory(
            CAM, model.model, model.layer, model.labels)


class torchcam_SmoothGradCAMpp(HeatmapHandler):
    def __init__(self, model: ModelHandler, interpolation=None, threshold=None):
        super().__init__(model, interpolation, threshold)
        self.execute = method_factory(
            SmoothGradCAMpp, model.model, model.layer, model.labels)


class torchcam_ScoreCAM(HeatmapHandler):
    def __init__(self, model: ModelHandler, interpolation=None, threshold=None):
        super().__init__(model, interpolation, threshold)
        self.execute = method_factory(
            ScoreCAM, model.model, model.layer, model.labels)


class torchcam_SSCAM(HeatmapHandler):
    def __init__(self, model: ModelHandler, interpolation=None, threshold=None):
        super().__init__(model, interpolation, threshold)
        self.execute = method_factory(
            SSCAM, model.model, model.layer, model.labels)


class torchcam_GradCam(HeatmapHandler):
    def __init__(self, model: ModelHandler, interpolation=None, threshold=None):
        super().__init__(model, interpolation, threshold)
        self.execute = method_factory(
            GradCAM, model.model, model.layer, model.labels)


class torchcam_GradCampp(HeatmapHandler):
    def __init__(self, model: ModelHandler, interpolation=None, threshold=None):
        super().__init__(model, interpolation, threshold)
        self.execute = method_factory(
            GradCAMpp, model.model, model.layer, model.labels)


class torchcam_XGradCAM(HeatmapHandler):
    def __init__(self, model: ModelHandler, interpolation=None, threshold=None):
        super().__init__(model, interpolation, threshold)
        self.execute = method_factory(
            XGradCAM, model.model, model.layer, model.labels)


class torchcam_LayerCAM(HeatmapHandler):
    def __init__(self, model: ModelHandler, interpolation=None, threshold=None):
        super().__init__(model, interpolation, threshold)
        self.execute = method_factory(
            LayerCAM, model.model, model.layer, model.labels)

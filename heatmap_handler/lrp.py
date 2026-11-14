import numpy as np
from model_handler.model_handler import ModelHandler
from .heatmap_handler import HeatmapHandler


# Class for simple Layer-wise Relevance Propagation (LRP) heatmap generation


def method_factory(model, layer, composite, canonizer, attributer=None):
    def generate_lrp_heatmap(prompt):

        def store_hook(module, input, output):
            if type(output) is tuple:
                output = output[-1]
            # set the current module's attribute 'output' to the its tensor
            module.output = output
            # keep the output tensor gradient, even if it is not a leaf-tensor
            output.retain_grad()

        img_tensor = prompt.original_image_tensor
        img_tensor.requires_grad = True
        composite_with_canonizer = None

        # Initialize the composite with the canonizer if provided

        if canonizer is None:
            composite_with_canonizer = composite()  # No canonizer
        else:
            try:
                composite_with_canonizer = composite(
                    canonizers=[canonizer()])
            except:
                composite_with_canonizer = composite(
                    canonizers=[canonizer(model.model)])

        # Use the composite rule with the specified canonizer in the context of the model
        grads = {}
        with composite_with_canonizer.context(model.model) as modified_model:
            handles = [module.register_forward_hook(
                store_hook) for module in modified_model.modules()]
            output = modified_model(img_tensor)
            prediction = output.max(1)[-1].item()
            output[0, prediction].backward()
            # for name, module in modified_model.named_modules():
            #     grads[name] = module.output.grad
            for name, module in modified_model.named_modules():
                grads[name] = module.output.grad

        for handle in handles:
            handle.remove()
        prompt.heatmap = grads[layer].squeeze(
            0).cpu().detach().mean(dim=0).numpy()
        prompt.predicted_class_id = prediction
        prompt.predicted_class_label = model.labels[prediction]
        return prompt

    return generate_lrp_heatmap


class HeatmapHandler:
    """Base class placeholder for HeatmapHandler."""

    def __init__(self, model: ModelHandler, interpolation=None, threshold=None):
        self.model = model
        self.interpolation = interpolation
        self.threshold = threshold


class PropagationHandler(HeatmapHandler):
    def __init__(self, model: ModelHandler, layer, composite, canonizer, attributer=None, interpolation=None, threshold=None):
        super().__init__(model, interpolation, threshold)  # Initialize the base class

        # Store the combination of layer, composite, canonizer, and attributer
        self.layer = layer
        self.composite = composite
        self.canonizer = canonizer
        self.attributer = attributer

        # Generate the execute method using the method factory
        self.execute = method_factory(
            model, layer, composite, canonizer, attributer)

        # Set instance name dynamically using class names
        self.instance_name = f"{self.layer}_" \
                             f"{str(self.composite)}_" \
                             f"{str(self.canonizer)}_" \
                             f"{'NoAttributer' if self.attributer is None else str(self.attributer)}"
        self.__class__.__name__ = self.instance_name

    def __call__(self, prompt):
        self.__class__.__name__ = self.instance_name
        # print(self.instance_name)
        if self.execute is None:
            raise NotImplementedError
        return self.execute(prompt)

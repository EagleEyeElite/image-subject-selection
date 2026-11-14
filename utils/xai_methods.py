import torch
import torch.nn.functional as F
from captum.attr import LayerGradCam

def store_hook(module, input, output):
    if type(output) is tuple:
        output = output[-1]
    # set the current module's attribute 'output' to the its tensor
    module.output = output
    # keep the output tensor gradient, even if it is not a leaf-tensor
    output.retain_grad()


class LRP:
    def __init__(
        self,
        model,
        composite,
        LRP_target_layer=None,
    ):
        self.model = model
        self.composite = composite
        self.LRP_target_layer = LRP_target_layer

    def __call__(self, img_tensor, pred):
        img_tensor.requires_grad = True
        img_tensor.grad = None
        with self.composite.context(self.model) as modified_model:
            if self.LRP_target_layer:
                handles = [
                    module.register_forward_hook(store_hook) for module in modified_model.modules()
                ]        
            
            output = modified_model(img_tensor)
            output[0, pred].backward()

     
            if self.LRP_target_layer is None:
                heatmap = img_tensor.grad.squeeze(0).cpu().detach().mean(dim=0)
            else:
                for name, module in modified_model.named_modules():
                    if name == self.LRP_target_layer:
                        heatmap = module.output.grad.squeeze(0).cpu().detach().mean(dim=0)
                        break
                for handle in handles:
                    handle.remove()

        return heatmap


class GradCAMHeatmap:
    def __init__(self, model, layer):
        self.layer_gc = LayerGradCam(model, layer)
        self.model = model

    def __call__(self, img_tensor, pred):

        heatmap = self.layer_gc.attribute(
            img_tensor, int(pred), relu_attributions=True
        ).detach()

        return heatmap.squeeze(0).squeeze(0)

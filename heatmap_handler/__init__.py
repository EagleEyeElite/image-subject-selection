
from functools import partial


def get_test_method():
    from .torchcam_imp import torchcam_LayerCAM
    return [torchcam_LayerCAM]


def get_all_methods():
    from .torchcam_imp import torchcam_CAM, torchcam_SmoothGradCAMpp, torchcam_ScoreCAM, torchcam_SSCAM, torchcam_GradCam, torchcam_GradCampp, torchcam_XGradCAM
    from .lrp import simpleLRP
    methods = [torchcam_CAM, torchcam_SmoothGradCAMpp, torchcam_ScoreCAM,
               torchcam_SSCAM, torchcam_GradCam, torchcam_GradCampp, torchcam_XGradCAM, simpleLRP]
    return methods


def get_specific_methods():
    from .torchcam_imp import torchcam_CAM, torchcam_SmoothGradCAMpp, torchcam_ScoreCAM, torchcam_SSCAM, torchcam_GradCam, torchcam_GradCampp, torchcam_XGradCAM
    from .lrp import simpleLRP
    methods = [torchcam_CAM,
               torchcam_GradCampp, torchcam_XGradCAM, simpleLRP, torchcam_SmoothGradCAMpp]
    return methods


def get_random_methods(n):
    from .torchcam_imp import torchcam_CAM, torchcam_SmoothGradCAMpp, torchcam_ScoreCAM, torchcam_SSCAM, torchcam_GradCam, torchcam_GradCampp, torchcam_XGradCAM
    import random
    methods = [torchcam_CAM, torchcam_SmoothGradCAMpp, torchcam_ScoreCAM,
               torchcam_SSCAM, torchcam_GradCam, torchcam_GradCampp, torchcam_XGradCAM]
    return random.sample(methods, n)


def get_lrp_methods():
    from .lrp import PropagationHandler
    from zennit.canonizers import SequentialMergeBatchNorm, NamedMergeBatchNorm, AttributeCanonizer, CompositeCanonizer
    from zennit.composites import EpsilonPlus, EpsilonAlpha2Beta1, EpsilonPlusFlat, EpsilonGammaBox, GuidedBackprop, ExcitationBackprop, EpsilonAlpha2Beta1Flat

    # layers = ["layer1", "layer2", "layer3", "layer4",
    #           "layer1.0", "layer1.0", "layer3.0", "layer4.0",
    #           "layer1.0.conv1", "layer1.0.conv1", "layer3.0.conv1", "layer4.0.conv1",
    #           "layer1.0.conv2", "layer1.0.conv2", "layer3.0.conv2", "layer4.0.conv2",
    #           "layer1.0.conv3", "layer1.0.conv3", "layer3.0.conv3", "layer4.0.conv3"]

    layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer1.0', 'layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.conv2', 'layer1.0.bn2', 'layer1.0.conv3', 'layer1.0.bn3', 'layer1.0.relu', 'layer1.0.downsample', 'layer1.0.downsample.0', 'layer1.0.downsample.1', 'layer1.1', 'layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.conv2', 'layer1.1.bn2', 'layer1.1.conv3', 'layer1.1.bn3', 'layer1.1.relu', 'layer1.2', 'layer1.2.conv1', 'layer1.2.bn1', 'layer1.2.conv2', 'layer1.2.bn2', 'layer1.2.conv3', 'layer1.2.bn3', 'layer1.2.relu', 'layer2', 'layer2.0', 'layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.conv2', 'layer2.0.bn2', 'layer2.0.conv3', 'layer2.0.bn3', 'layer2.0.relu', 'layer2.0.downsample', 'layer2.0.downsample.0', 'layer2.0.downsample.1', 'layer2.1', 'layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.conv2', 'layer2.1.bn2', 'layer2.1.conv3', 'layer2.1.bn3', 'layer2.1.relu', 'layer2.2', 'layer2.2.conv1', 'layer2.2.bn1', 'layer2.2.conv2', 'layer2.2.bn2', 'layer2.2.conv3', 'layer2.2.bn3', 'layer2.2.relu', 'layer2.3', 'layer2.3.conv1', 'layer2.3.bn1', 'layer2.3.conv2', 'layer2.3.bn2', 'layer2.3.conv3', 'layer2.3.bn3', 'layer2.3.relu', 'layer3', 'layer3.0', 'layer3.0.conv1', 'layer3.0.bn1', 'layer3.0.conv2', 'layer3.0.bn2', 'layer3.0.conv3', 'layer3.0.bn3',
              'layer3.0.relu', 'layer3.0.downsample', 'layer3.0.downsample.0', 'layer3.0.downsample.1', 'layer3.1', 'layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.conv2', 'layer3.1.bn2', 'layer3.1.conv3', 'layer3.1.bn3', 'layer3.1.relu', 'layer3.2', 'layer3.2.conv1', 'layer3.2.bn1', 'layer3.2.conv2', 'layer3.2.bn2', 'layer3.2.conv3', 'layer3.2.bn3', 'layer3.2.relu', 'layer3.3', 'layer3.3.conv1', 'layer3.3.bn1', 'layer3.3.conv2', 'layer3.3.bn2', 'layer3.3.conv3', 'layer3.3.bn3', 'layer3.3.relu', 'layer3.4', 'layer3.4.conv1', 'layer3.4.bn1', 'layer3.4.conv2', 'layer3.4.bn2', 'layer3.4.conv3', 'layer3.4.bn3', 'layer3.4.relu', 'layer3.5', 'layer3.5.conv1', 'layer3.5.bn1', 'layer3.5.conv2', 'layer3.5.bn2', 'layer3.5.conv3', 'layer3.5.bn3', 'layer3.5.relu', 'layer4', 'layer4.0', 'layer4.0.conv1', 'layer4.0.bn1', 'layer4.0.conv2', 'layer4.0.bn2', 'layer4.0.conv3', 'layer4.0.bn3', 'layer4.0.relu', 'layer4.0.downsample', 'layer4.0.downsample.0', 'layer4.0.downsample.1', 'layer4.1', 'layer4.1.conv1', 'layer4.1.bn1', 'layer4.1.conv2', 'layer4.1.bn2', 'layer4.1.conv3', 'layer4.1.bn3', 'layer4.1.relu', 'layer4.2', 'layer4.2.conv1', 'layer4.2.bn1', 'layer4.2.conv2', 'layer4.2.bn2', 'layer4.2.conv3', 'layer4.2.bn3', 'layer4.2.relu', ]
    # layers = ['conv1', 'bn1', 'relu', ]
    composites = [EpsilonPlus, EpsilonAlpha2Beta1, EpsilonPlusFlat,
                  GuidedBackprop, ExcitationBackprop, EpsilonAlpha2Beta1Flat]

    # EpsilonGammaBox needs parameters
    # low and high values for ZBox need to be Tensors in the shape of the input
    # the batch-dimension may be chosen larger, to support different sizes
    # composite = EpsilonGammaBox(
    #     low=-3 * sigma,
    #     high=3 * sigma,
    # )

    canonizers = [None, SequentialMergeBatchNorm]  # rest need parameters

    # composites = [EpsilonPlus, EpsilonAlpha2Beta1, EpsilonPlusFlat, EpsilonGammaBox,
    #               GuidedBackprop, ExcitationBackprop, EpsilonAlpha2Beta1Flat]
    # canonizers = [None, SequentialMergeBatchNorm, NamedMergeBatchNorm,
    #               AttributeCanonizer, CompositeCanonizer]

    all_methods = []
    for layer in layers:
        for composite in composites:
            for canonizer in canonizers:
                # Use default arguments in lambda to capture current values of `layer`, `composite`, and `canonizer`
                all_methods.append(lambda x, l=layer, c=composite,
                                   ca=canonizer: PropagationHandler(x, l, c, ca))
    print(len(all_methods))
    return all_methods

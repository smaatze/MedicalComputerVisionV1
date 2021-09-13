
import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .unet_2d import *

def get_network(name, n_classes, in_channels=3, feature_scale=4, tensor_dim='2d'):

    if name == 'unet':
        model = get_model_instance(name, tensor_dim)
        model = model(n_classes=n_classes,
                      feature_scale=feature_scale,
                      in_channels=in_channels,
                      is_deconv=False,
                      is_batchnorm = True)
    elif name == 'maskrcnn':
        model = get_maskrcnn_instance(n_classes=n_classes)

    return model

def get_model_instance(name, tensor_dim):
    return {
        'unet': {'2d': Unet2d}
        # 'atrous_crf':  {'2d': atrous_crf}
    }[name][tensor_dim]

def get_maskrcnn_instance(n_classes=2):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes=n_classes)
    return model
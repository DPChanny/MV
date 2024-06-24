import json
import os

from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2, fasterrcnn_resnet50_fpn, faster_rcnn)

from configs import PROJECT_PATH
from CoordConv2d import CoordConv2d
from vlc_detector_configs import VLCDetectorVersion, CoordConv2dVersion


def collate_fn(batch):
    image_list = []
    target_list = []
    for image, target in batch:
        image_list.append(image)
        target_list.append(target)

    return image_list, target_list


def get_model(model_version, coord_conv_2d_version, device):
    if model_version == VLCDetectorVersion.V1_PRETRAINED:
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                                        tranable_layers=5)
    elif model_version == VLCDetectorVersion.V2_PRETRAINED:
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                                           tranable_layers=5)
    elif model_version == VLCDetectorVersion.V1:
        model = fasterrcnn_resnet50_fpn()
    else:
        model = fasterrcnn_resnet50_fpn_v2()

    num_classes = len(get_vlc_map()) + 1
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    if coord_conv_2d_version is not CoordConv2dVersion.NONE:
        model.backbone.body.conv1 = CoordConv2d(model.backbone.body.conv1)

    model.to(device)
    print(model)

    return model


def get_vlc_map():
    with open(os.path.join(PROJECT_PATH, "vlc_map.json"), "r") as file:
        data = json.load(file)

    return data

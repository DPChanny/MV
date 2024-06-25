import os

import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2, fasterrcnn_resnet50_fpn, faster_rcnn)

from utils import get_vlc_map
from img2vlc.CoordConv2d import CoordConv2d
from img2vlc.img2vlc_configs import (ModelVersion, CoordConv2dVersion,
                                     MODEL_PATH, COORD_CONV_2D_VERSION, IMG2VLC_VERSION)


def collate_fn(batch):
    image_list = []
    target_list = []
    for image, target in batch:
        image_list.append(image)
        target_list.append(target)

    return image_list, target_list


def get_model(model_version, coord_conv_2d_version, device, log_model=True):
    if model_version == ModelVersion.V1_PRETRAINED:
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                                        tranable_layers=5)
    elif model_version == ModelVersion.V2_PRETRAINED:
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                                           tranable_layers=5)
    elif model_version == ModelVersion.V1:
        model = fasterrcnn_resnet50_fpn()
    else:
        model = fasterrcnn_resnet50_fpn_v2()

    num_classes = len(get_vlc_map()) + 1
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    if coord_conv_2d_version is not CoordConv2dVersion.NONE:
        model.backbone.body.conv1 = CoordConv2d(model.backbone.body.conv1)

    model.to(device)
    if log_model:
        print(model)

    return model


def load_checkpoint(device):
    if os.path.exists(os.path.join(MODEL_PATH,
                                   str(IMG2VLC_VERSION),
                                   str(COORD_CONV_2D_VERSION),
                                   "checkpoint.pth")):
        checkpoint = torch.load(os.path.join(MODEL_PATH,
                                             str(IMG2VLC_VERSION),
                                             str(COORD_CONV_2D_VERSION),
                                             "checkpoint.pth"),
                                map_location=device)
    else:
        checkpoint = None

    return checkpoint


def load_optimizer(optimizer, checkpoint):
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return optimizer


def load_starts(checkpoint):
    if checkpoint is not None:
        start_epoch = checkpoint['start_epoch']
        start_batch = checkpoint['start_batch']
    else:
        start_epoch = 0
        start_batch = 0

    return start_epoch, start_batch


def load_model(model, checkpoint):
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])

    return model


def load_scheduler(scheduler, checkpoint):
    if checkpoint is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return scheduler


def save_checkpoint(epoch, total_epoch, batch, total_batch, model, optimizer, scheduler):
    if not os.path.exists(os.path.join(MODEL_PATH,
                                       str(IMG2VLC_VERSION),
                                       str(COORD_CONV_2D_VERSION))):
        os.makedirs(os.path.join(MODEL_PATH,
                                 str(IMG2VLC_VERSION),
                                 str(COORD_CONV_2D_VERSION)))

    values = {'start_epoch': epoch,
              'start_batch': batch,
              'model': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'scheduler': scheduler.state_dict()}

    torch.save(values, os.path.join(MODEL_PATH,
                                    str(IMG2VLC_VERSION),
                                    str(COORD_CONV_2D_VERSION),
                                    "{}-{}_{}-{}.pth".format(epoch, total_epoch,
                                                             batch, total_batch)))
    torch.save(values, os.path.join(MODEL_PATH,
                                    str(IMG2VLC_VERSION),
                                    str(COORD_CONV_2D_VERSION),
                                    "checkpoint.pth".format(epoch, total_epoch,
                                                            batch, total_batch)))

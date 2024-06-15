import json

import cv2
import torchvision
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

RAW_DATA_PATH: str = ".\\raw_data"
DATA_PATH: str = ".\\data"

JSON_PATH: str = "jsons"
JPG_PATH: str = "jpgs"


def json_parser(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)

    return (data['visible_latex_chars'],
            [(data['x_mins'][i], data['y_mins'][i],
              data['x_maxes'][i], data['y_maxes'][i])
             for i in range(len(data['visible_latex_chars']))])


def get_visible_latex_char_map():
    with open(".\\visible_latex_char_map.json", "r") as file:
        data = json.load(file)

    return data


def get_image(image, prediction, target, iou_threshold):
    alpha = 1
    image = image.copy()
    cv2.putText(image, "{} x {} ({:.3f} ~ {:.3f})".format(image.shape[0], image.shape[1],
                                                          iou_threshold[0], iou_threshold[1]),
                (0, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 0, 255), 1)

    if target is not None:
        for index, visible_latex_char in enumerate(target['visible_latex_chars']):
            overlay = image.copy()
            cv2.rectangle(overlay,
                          (int(target['boxes'][index][0]), int(target['boxes'][index][1])),
                          (int(target['boxes'][index][2]), int(target['boxes'][index][3])),
                          color=(255, 0, 0, 255),
                          thickness=1)
            cv2.putText(overlay, visible_latex_char,
                        (int(target['boxes'][index][0]), int(target['boxes'][index][1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0, 255), 1)
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    for index, visible_latex_char in enumerate(prediction['visible_latex_chars']):
        if prediction['scores'][index] < iou_threshold[0] or prediction['scores'][index] > iou_threshold[1]:
            continue
        overlay = image.copy()
        cv2.rectangle(overlay,
                      (int(prediction['boxes'][index][0]), int(prediction['boxes'][index][1])),
                      (int(prediction['boxes'][index][2]), int(prediction['boxes'][index][3])),
                      color=(0, 255, 0, 255),
                      thickness=1)
        cv2.putText(overlay, visible_latex_char + " ({:.3f})".format(prediction['scores'][index]),
                    (int(prediction['boxes'][index][0]), int(prediction['boxes'][index][1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0, 255), 1)
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return image


def visualize(image_list, prediction_list, target_list=None):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    iou_threshold = [0.0, 1.0]
    iou_threshold_sensitivity = 0.025
    image_index = 0

    while True:
        cv2.imshow('image', get_image(image_list[image_index],
                                      prediction_list[image_index],
                                      target_list[image_index] if target_list is not None else None,
                                      iou_threshold))
        key = cv2.waitKey(0)

        if key == ord('Q') or key == ord('q'):
            break
        elif key == ord('['):
            iou_threshold[0] = max(0.0, iou_threshold[0] - iou_threshold_sensitivity)
        elif key == ord(']'):
            iou_threshold[0] = min(iou_threshold[1], iou_threshold[0] + iou_threshold_sensitivity)
        elif key == ord('{'):
            iou_threshold[1] = max(iou_threshold[0], iou_threshold[1] - iou_threshold_sensitivity)
        elif key == ord('}'):
            iou_threshold[1] = min(1.0, iou_threshold[1] + iou_threshold_sensitivity)
        elif key == ord(',') or key == ord('<'):
            image_index = max(image_index - 1, 0)
        elif key == ord('.') or key == ord('>'):
            image_index = min(image_index + 1, len(image_list) - 1)


def collate_fn(batch):
    image_list = []
    target_list = []
    for image, target in batch:
        image_list.append(image)
        target_list.append(target)

    return image_list, target_list


def get_model(device):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                                    trainable_backbone_layers=5,
                                    weights_backbone=ResNet50_Weights.DEFAULT)

    num_classes = len(get_visible_latex_char_map()) + 1
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    model.to(device)

    return model

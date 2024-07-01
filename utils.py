import json
import os
import time

import cv2

from configs import PROJECT_PATH


def json_parser(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)

    return (data['flcs'], data['vlcs'],
            [(x_min, y_min, x_max, y_max) for x_min, y_min, x_max, y_max
             in zip(data['x_mins'], data['y_mins'], data['x_maxs'], data['y_maxs'])])


def get_image(image, prediction, target, iou_threshold):
    alpha = 1
    image = image.copy()
    if target is not None:
        for vlc, box in zip(target['vlcs'], target['boxes']):
            overlay = image.copy()
            cv2.rectangle(overlay,
                          (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          color=(255, 0, 0, 255), thickness=1)
            cv2.putText(overlay, vlc, (int(box[0]), int(box[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0, 255), 1)
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    for vlc, score, box in zip(prediction['vlcs'], prediction['scores'], prediction['boxes']):
        if score < iou_threshold[0] or score > iou_threshold[1]:
            continue
        overlay = image.copy()
        cv2.rectangle(overlay, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                      color=(0, 255, 0, 255), thickness=1)
        cv2.putText(overlay, vlc + " ({:.3f})".format(score), (int(box[0]), int(box[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0, 255), 1)
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return image


def visualize(image_list, prediction_list, target_list=None):
    cv2.namedWindow('window', cv2.WINDOW_AUTOSIZE)

    iou_threshold = [0.0, 1.0]
    iou_threshold_sensitivity = 0.025
    image_index = 0

    while True:
        cv2.imshow('window', get_image(image_list[image_index],
                                       prediction_list[image_index],
                                       target_list[image_index] if target_list is not None else None,
                                       iou_threshold))

        cv2.setWindowTitle('window', "Prediction Result({}/{}): {} x {} ({:.3f} ~ {:.3f})".format(
            image_index + 1, len(image_list),
            image_list[image_index].shape[0], image_list[image_index].shape[1],
            iou_threshold[0], iou_threshold[1]))

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


class Timer:
    def __init__(self):
        self.last_start = time.time()

    def end(self):
        elapsed_time = time.time() - self.last_start
        self.last_start = time.time()
        return elapsed_time

    def start(self):
        self.last_start = time.time()


def get_vlc2tok():
    with open(os.path.join(PROJECT_PATH, "vlc2tok.json"), "r") as file:
        vlc2tok = json.load(file)

    return vlc2tok


def get_tok2vlc():
    return {v: k for k, v in get_vlc2tok().items()}


def get_flc2tok():
    vlc2tok = get_vlc2tok()

    vlc2tok['{'] = len(vlc2tok) + 1
    vlc2tok['}'] = len(vlc2tok) + 1
    vlc2tok['['] = len(vlc2tok) + 1
    vlc2tok[']'] = len(vlc2tok) + 1
    vlc2tok['^'] = len(vlc2tok) + 1
    vlc2tok['_'] = len(vlc2tok) + 1
    vlc2tok['<PAD>'] = len(vlc2tok) + 1
    vlc2tok['<SOS>'] = len(vlc2tok) + 1
    vlc2tok['<EOS>'] = len(vlc2tok) + 1

    return vlc2tok


def get_tok2flc():
    return {v: k for k, v in get_flc2tok().items()}


_flc2tok = get_flc2tok()

PAD_INDEX = _flc2tok['<PAD>']
SOS_INDEX = _flc2tok['<SOS>']
EOS_INDEX = _flc2tok['<EOS>']

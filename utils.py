import json
import time

import cv2


def json_parser(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)

    return (data['vlcs'],
            [(data['x_mins'][vlc_index], data['y_mins'][vlc_index],
              data['x_maxes'][vlc_index], data['y_maxes'][vlc_index])
             for vlc_index in range(len(data['vlcs']))])


def get_image(image, prediction, target, iou_threshold):
    alpha = 1
    image = image.copy()
    cv2.putText(image, "{} x {} ({:.3f} ~ {:.3f})".format(image.shape[0], image.shape[1],
                                                          iou_threshold[0], iou_threshold[1]),
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0, 255), 1)

    if target is not None:
        for vlc_index in target['vlcs']:
            overlay = image.copy()
            cv2.rectangle(overlay,
                          (int(target['boxes'][vlc_index][0]), int(target['boxes'][vlc_index][1])),
                          (int(target['boxes'][vlc_index][2]), int(target['boxes'][vlc_index][3])),
                          color=(255, 0, 0, 255),
                          thickness=1)
            cv2.putText(overlay, target['vlcs'][vlc_index],
                        (int(target['boxes'][vlc_index][0]), int(target['boxes'][vlc_index][1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0, 255), 1)
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    for vlc_index in prediction['vlcs']:
        if prediction['scores'][vlc_index] < iou_threshold[0] or prediction['scores'][vlc_index] > iou_threshold[1]:
            continue
        overlay = image.copy()
        cv2.rectangle(overlay,
                      (int(prediction['boxes'][vlc_index][0]), int(prediction['boxes'][vlc_index][1])),
                      (int(prediction['boxes'][vlc_index][2]), int(prediction['boxes'][vlc_index][3])),
                      color=(0, 255, 0, 255),
                      thickness=1)
        cv2.putText(overlay, prediction['vlcs'][vlc_index] + " ({:.3f})".format(prediction['scores'][vlc_index]),
                    (int(prediction['boxes'][vlc_index][0]), int(prediction['boxes'][vlc_index][1]) - 5),
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


class Timer:
    def __init__(self):
        self.last_start = time.time()

    def end(self):
        elapsed_time = time.time() - self.last_start
        self.last_start = time.time()
        return elapsed_time

    def start(self):
        self.last_start = time.time()

import json

import cv2

RAW_DATA_PATH = ".\\raw_data"
DATA_PATH = ".\\data"

JSON_PATH = "jsons"
JPG_PATH = "jpgs"


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


def get_visible_latex_char_color_map():
    with open("visible_latex_char_color_map.json", "r") as file:
        data = json.load(file)

    return data


def draw_boxes(image, boxes, visible_latex_chars):
    visible_latex_char_color_map = get_visible_latex_char_color_map()
    image = image.copy()
    for index, visible_latex_char in visible_latex_chars:
        cv2.rectangle(image,
                      (int(boxes[index][0]), int(boxes[index][1])),
                      (int(boxes[index][2]), int(boxes[index][3])),
                      color=visible_latex_char_color_map[visible_latex_char],
                      thickness=1)
        cv2.putText(image, visible_latex_char,
                    (int(boxes[index][0]), int(boxes[index][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, visible_latex_char_color_map[visible_latex_char], 2)
    return image

import os

from PIL import Image

from configs import JSON_PATH, DATA_PATH, JPG_PATH
from utils import json_parser

json_list = [file for file in os.listdir(os.path.join(DATA_PATH, JSON_PATH))
             if file.endswith(".json")]

vlc_lens = []
flc_lens = []
widths = []
heights = []

for index, json in enumerate(json_list):
    if not index % 1000:
        print(index)
    data_name = os.path.splitext(json)[0]
    flcs, vlcs, boxes = json_parser(os.path.join(DATA_PATH,
                                                 JSON_PATH,
                                                 json))

    vlc_lens.append(len(vlcs))
    flc_lens.append(len(flcs))

    image = Image.open(os.path.join(DATA_PATH, JPG_PATH, data_name + ".jpg")).convert("RGB")
    widths.append(image.width)
    heights.append(image.height)
    image.close()

print(max(flc_lens), min(flc_lens), sum(flc_lens) / len(flc_lens))
print(max(vlc_lens), min(vlc_lens), sum(vlc_lens) / len(vlc_lens))
print(max(widths), min(widths), sum(widths) / len(widths))
print(max(heights), min(heights), sum(heights) / len(heights))

# 115 7 34.30829
# 64 5 18.56227
# 5340 259 1401.68117
# 1197 100 376.78104

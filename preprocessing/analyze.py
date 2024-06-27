import os
import json

from PIL import Image

from configs import JSON_PATH, DATA_PATH, JPG_PATH
from utils import json_parser

import matplotlib.pyplot as plt

json_list = [file for file in os.listdir(os.path.join(DATA_PATH, JSON_PATH))
             if file.endswith(".json")]

from_start = False

result = {'vlc_count': {},
          'vlc_lens': [],
          'flc_lens': [],
          'widths': [],
          'heights': []}

if from_start:

    for index, json in enumerate(json_list):
        if not index % 1000:
            print(index)
        data_name = os.path.splitext(json)[0]
        flcs, vlcs, boxes = json_parser(os.path.join(DATA_PATH,
                                                     JSON_PATH,
                                                     json))

        result['vlc_lens'].append(len(vlcs))
        result['flc_lens'].append(len(flcs))
        for vlc in vlcs:
            if vlc not in result['vlc_count']:
                result['vlc_count'][vlc] = 0
            result['vlc_count'][vlc] += 1

        image = Image.open(os.path.join(DATA_PATH, JPG_PATH, data_name + ".jpg")).convert("RGB")
        result['widths'].append(image.width)
        result['heights'].append(image.height)
        image.close()

    print(max(result['flc_lens']), min(result['flc_lens']), sum(result['flc_lens']) / len(result['flc_lens']))
    print(max(result['vlc_lens']), min(result['vlc_lens']), sum(result['vlc_lens']) / len(result['vlc_lens']))
    print(max(result['widths']), min(result['widths']), sum(result['widths']) / len(result['widths']))
    print(max(result['heights']), min(result['heights']), sum(result['heights']) / len(result['heights']))
    print(result['vlc_count'])

    with open("result.json", "w") as f:
        json.dump(result, f)

with open("result.json", "r") as f:
    result = json.load(f)

plt.bar(result['vlc_count'].keys(), result['vlc_count'].values())
plt.xticks(result['vlc_count'].keys(), result['vlc_count'].values())
plt.imsave("result.png")


# 115 7 34.30829
# 64 5 18.56227
# 5340 259 1401.68117
# 1197 100 376.78104

# {'\\lim_': 111223, 'v': 15460, '\\to': 111223, '3': 62734, '\\frac': 140649, 'd': 44148, '\\left(': 54150,
# 'e': 6420, '+': 113617, '-': 95911, '2': 91587, '\\sin': 39494, '9': 42619, '\\right)': 54150, '1': 35085,
# '8': 41951, '4': 42610, 'c': 15571, '\\pi': 43990, '\\cos': 28729, '\\tan': 34033, '=': 9801, 't': 26317,
# '/': 25945, '\\sec': 11581, 'x': 63050, '7': 41773, '0': 19159, '\\cot': 3080, 'h': 31189, '\\ln': 10980,
# 'w': 30188, '\\infty': 21187, 'k': 15634, '6': 41671, 'y': 29048, '5': 41726, 'u': 30337, '\\sqrt': 12140,
# 's': 14038, '\\log': 15222, 'z': 16233, '\\cdot': 1264, '\\csc': 6827, '\\left|': 4407, '\\right|': 4407,
# 'p': 15474, 'b': 15527, 'n': 11906, 'g': 15613, 'a': 15615, '\\theta': 14726, 'r': 14753, '.': 55}

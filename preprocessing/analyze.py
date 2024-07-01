import json
import os

from PIL import Image
from matplotlib.gridspec import GridSpec

from configs import JSON_PATH, DATA_PATH, JPG_PATH
from utils import json_parser, get_flc2tok

import matplotlib.pyplot as plt

json_list = [file for file in os.listdir(os.path.join(DATA_PATH, JSON_PATH))
             if file.endswith(".json")]

from_start = False

result = {'vlc_count': {},
          'vlc_lens': [],
          'flc_lens': [],
          'widths': [],
          'heights': [],
          'sizes': []}

if from_start:
    for index, json_path in enumerate(json_list):
        if not index % 1000:
            print(index)
        data_name = os.path.splitext(json_path)[0]
        flcs, vlcs, boxes = json_parser(os.path.join(DATA_PATH,
                                                     JSON_PATH,
                                                     json_path))

        result['vlc_lens'].append(len(vlcs))
        result['flc_lens'].append(len(flcs))
        for vlc in vlcs:
            if vlc not in result['vlc_count']:
                result['vlc_count'][vlc] = 0
            result['vlc_count'][vlc] += 1

        image = Image.open(os.path.join(DATA_PATH, JPG_PATH, data_name + ".jpg")).convert("RGB")
        result['widths'].append(image.width)
        result['heights'].append(image.height)
        result['sizes'].append(image.width * image.height)

        image.close()

    with open("result.json", "w") as f:
        json.dump(result, f)

with open("result.json", "r") as f:
    result = json.load(f)

fig = plt.figure(figsize=(40, 20))
gs = GridSpec(nrows=2, ncols=3)

for vlc in get_flc2tok().keys():
    if vlc not in result['vlc_count']:
        result['vlc_count'][vlc] = 0
result['vlc_count'] = {k: v for (k, v) in sorted(result['vlc_count'].items(), key=lambda x: x[1], reverse=True)}

ax1 = fig.add_subplot(gs[0, :])
ax1.set_title('vlc count')
ax1.bar(result['vlc_count'].keys(), result['vlc_count'].values())
ax1.set_xticks(range(len(result['vlc_count'].keys())))
ax1.set_xticklabels(result['vlc_count'].keys(), rotation=90)

ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title('vlc lens')
ax2.plot(result['vlc_lens'], linewidth=0.01)
ax2.hlines(sum(result['vlc_lens']) / len(result['vlc_lens']), 0, len(result['vlc_lens']) - 1, color='red')

ax3 = fig.add_subplot(gs[1, 1])
ax3.set_title('flc lens')
ax3.plot(result['flc_lens'], linewidth=0.01)
ax3.hlines(sum(result['flc_lens']) / len(result['flc_lens']), 0, len(result['flc_lens']) - 1, color='red')

ax4 = fig.add_subplot(gs[1, 2])
ax4.set_title('sizes')
ax4.plot(result['sizes'], linewidth=0.01)
ax4.hlines(sum(result['sizes']) / len(result['sizes']), 0, len(result['sizes']) - 1, color='red')

fig.show()
fig.savefig("result.png")

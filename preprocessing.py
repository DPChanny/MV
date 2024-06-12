import json
import os
from PIL import Image

from utils import JSON_PATH, DATA_PATH, JPG_PATH, RAW_DATA_PATH

if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

if not os.path.exists(os.path.join(DATA_PATH, JSON_PATH)):
    os.mkdir(os.path.join(DATA_PATH, JSON_PATH))
if not os.path.exists(os.path.join(DATA_PATH, JPG_PATH)):
    os.mkdir(os.path.join(DATA_PATH, JPG_PATH))

data_index = 0

for batch in os.listdir(RAW_DATA_PATH):
    with open(os.path.join(
            RAW_DATA_PATH,
            batch,
            "JSON",
            "kaggle_data_" + batch.split('_')[1] + ".json"), "r") as raw_file:

        batch_data_map = {}
        for batch_data_info in json.load(raw_file):
            batch_data_map[batch_data_info['filename']] = batch_data_info

        batch_data_count = len(os.listdir(os.path.join(RAW_DATA_PATH, batch, "background_images")))

        for batch_data_index, batch_data in enumerate(os.listdir(os.path.join(RAW_DATA_PATH, batch, "background_images"))):
            if not batch_data_index % 100:
                print(batch_data, data_index, str(batch_data_index) + "/" + str(batch_data_count) + " of " + batch)

            data = {'width': batch_data_map[batch_data]['image_data']['width'],
                    'height': batch_data_map[batch_data]['image_data']['height'],
                    'depth': batch_data_map[batch_data]['image_data']['depth'],
                    'full_latex_chars': batch_data_map[batch_data]['image_data']['full_latex_chars'],
                    'visible_latex_chars': batch_data_map[batch_data]['image_data']['visible_latex_chars'],
                    'x_mins': batch_data_map[batch_data]['image_data']['xmins_raw'],
                    'x_maxes': batch_data_map[batch_data]['image_data']['xmaxs_raw'],
                    'y_mins': batch_data_map[batch_data]['image_data']['ymins_raw'],
                    'y_maxes': batch_data_map[batch_data]['image_data']['ymaxs_raw']}

            with open(os.path.join(DATA_PATH, JSON_PATH, str(data_index) + ".json"), "w") as file:
                json.dump(data, file)

            image = Image.open(os.path.join(RAW_DATA_PATH, batch, "background_images", batch_data))
            image.save(os.path.join(DATA_PATH, JPG_PATH, str(data_index) + ".jpg"), "JPEG")

            data_index += 1

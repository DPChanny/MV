import os.path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import json_parser, get_visible_latex_char_map, JSON_PATH, JPG_PATH


class MVDataset(Dataset):
    def __init__(self, data_path: str, device, transforms):
        self.data_path = data_path
        self.json_list = [file for file in os.listdir(os.path.join(data_path, JSON_PATH)) if file.endswith(".json")]
        self.device = device
        self.transforms = transforms

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, data_index):
        data_name = os.path.splitext(self.json_list[data_index])[0]
        visible_latex_chars, boxes = json_parser(os.path.join(self.data_path,
                                                              JSON_PATH,
                                                              self.json_list[data_index]))

        image = Image.open(os.path.join(self.data_path, JPG_PATH, data_name + ".jpg")).convert("RGB")

        boxes = torch.tensor(boxes, device=self.device, dtype=torch.float32)

        visible_latex_char_map = get_visible_latex_char_map()

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {'boxes': boxes,
                  'labels': torch.tensor([visible_latex_char_map[i] for i in visible_latex_chars],
                                         dtype=torch.int64,
                                         device=self.device),
                  'area': area,
                  'image_id': data_index}

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, [target]

import os.path

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as tt

from configs import JSON_PATH, JPG_PATH
from utils import json_parser
from vlc_detector.utils import get_vlc_map


class MVDataset(Dataset):
    def __init__(self, data_path, json_list, device, is_train):
        self.data_path = data_path
        self.json_list = json_list
        self.device = device
        transforms = [tt.PILToTensor(),
                      tt.ToDtype(torch.float32, scale=True)]
        if is_train:
            transforms.append(tt.RandomHorizontalFlip(0.5))
        self.transforms = tt.Compose(transforms)

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, data_index):
        data_name = os.path.splitext(self.json_list[data_index])[0]
        vlcs, boxes = json_parser(os.path.join(self.data_path,
                                               JSON_PATH,
                                               self.json_list[data_index]))

        image = Image.open(os.path.join(self.data_path, JPG_PATH, data_name + ".jpg")).convert("RGB")

        boxes = torch.tensor(boxes, device=self.device, dtype=torch.float32)

        vlc_map = get_vlc_map()

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {'boxes': boxes,
                  'labels': torch.tensor([vlc_map[i] for i in vlcs],
                                         dtype=torch.int64,
                                         device=self.device),
                  'area': area,
                  'image_id': data_index}

        image, target = self.transforms(image, target)
        image = image.to(self.device)

        return image, target

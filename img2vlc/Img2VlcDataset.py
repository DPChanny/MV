import os.path

import torch
import torchvision.transforms.v2 as tt
from PIL import Image
from torch.utils.data import Dataset

from configs import JSON_PATH, JPG_PATH
from utils import json_parser, get_flc2tok


class Img2VlcDataset(Dataset):
    def __init__(self, data_path, json_list, device, is_train):
        self.data_path = data_path
        self.json_list = json_list
        self.device = device
        transforms = [tt.PILToTensor(),
                      tt.ToDtype(torch.float32, scale=True)]
        if is_train:
            transforms.append(tt.RandomHorizontalFlip(0.5))
        self.transforms = tt.Compose(transforms)
        self.vlc2tok = get_flc2tok()

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, data_index):
        data_name = os.path.splitext(self.json_list[data_index])[0]
        _, vlcs, boxes = json_parser(os.path.join(self.data_path,
                                                  JSON_PATH,
                                                  self.json_list[data_index]))

        image = Image.open(os.path.join(self.data_path, JPG_PATH, data_name + ".jpg")).convert("RGB")

        boxes = torch.tensor(boxes).to(self.device).type(torch.float32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {'boxes': boxes,
                  'labels': torch.tensor([self.vlc2tok[i] for i in vlcs]).to(self.device).type(torch.int64),
                  'area': area,
                  'image_id': data_index}

        image, target = self.transforms(image, target)
        image = image.to(self.device)

        return image, target

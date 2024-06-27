import os.path

import torch
import torchvision.transforms.v2 as tt
from PIL import Image
from torch.utils.data import Dataset

from configs import JSON_PATH, JPG_PATH
from utils import json_parser
from img2vlc.img2vlc_utils import get_vlc2tok


class Vlc2FlcDataset(Dataset):
    def __init__(self, data_path, json_list, device, is_train):
        self.data_path = data_path
        self.json_list = json_list
        self.device = device
        self.vlc2tok = get_vlc2tok()

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, data_index):
        data_name = os.path.splitext(self.json_list[data_index])[0]
        flcs, vlcs, _ = json_parser(os.path.join(self.data_path,
                                                 JSON_PATH,
                                                 self.json_list[data_index]))



        image, target = self.transforms(image, target)
        image = image.to(self.device)

        return image, target

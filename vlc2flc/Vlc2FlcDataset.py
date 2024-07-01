import os.path

import torch
from torch.utils.data import Dataset

from configs import JSON_PATH
from utils import json_parser, get_flc2tok
from vlc2flc.vlc2flc_utils import tensor_transform


class Vlc2FlcDataset(Dataset):
    def __init__(self, data_path, json_list, device):
        self.data_path = data_path
        self.json_list = json_list
        self.device = device
        self.flc2tok = get_flc2tok()

    def __len__(self):
        return len(self.json_list)

    def __getitem__(self, data_index):
        flcs, vlcs, boxes = json_parser(os.path.join(self.data_path, JSON_PATH, self.json_list[data_index]))
        src = tensor_transform([self.flc2tok[vlc] for vlc in vlcs], self.device)
        tgt = tensor_transform([self.flc2tok[flc] for flc in flcs], self.device)
        boxes = torch.tensor(boxes).to(self.device).type(torch.int32)

        return src, tgt, boxes

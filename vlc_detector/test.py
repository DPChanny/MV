import os
import random

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms.v2.functional import to_pil_image

from MVDataset import MVDataset
from configs import DATA_PATH, JSON_PATH
from utils import visualize
from vlc_detector_configs import MODEL_PATH, VLC_DETECTOR_VERSION, COORD_CONV_2D_VERSION
from vlc_detector_utils import get_vlc_map, get_model, collate_fn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = get_model(VLC_DETECTOR_VERSION, COORD_CONV_2D_VERSION, device)

model.load_state_dict(torch.load(os.path.join(MODEL_PATH,
                                              str(VLC_DETECTOR_VERSION),
                                              str(COORD_CONV_2D_VERSION),
                                              "check_point.pth"),
                                 map_location=device)['model'])
model.eval()

test_data_count = 5

json_list = random.sample([file
                           for file in os.listdir(os.path.join(DATA_PATH, JSON_PATH))
                           if file.endswith(".json")], test_data_count)

dataset = MVDataset(DATA_PATH, json_list, device, False)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

vlc_map = get_vlc_map()
vlc_map = {v: k for k, v in vlc_map.items()}

prediction_list = []
image_list = []
target_list = []

images: Tensor
for test_data_index, (images, targets) in enumerate(dataloader):
    print("PROGRESS {:.2f}%".format(test_data_index / len(dataloader) * 100))

    tensor_prediction_list = model(images)

    for image, prediction, target in zip(images, tensor_prediction_list, targets):
        prediction_list.append({
            'boxes': prediction['boxes'].cpu().detach().numpy(),
            'scores': prediction['scores'].cpu().detach().numpy(),
            'vlcs': [vlc_map[label]
                     for label in prediction['labels'].cpu().detach().numpy()]
        })
        image_list.append(cv2.cvtColor(np.array(to_pil_image(images)), cv2.COLOR_RGB2BGR))
        target_list.append({
            'boxes': target['boxes'].cpu().detach().numpy(),
            'vlcs': [vlc_map[label]
                     for label in target['labels'].cpu().detach().numpy()]
        })

visualize(image_list, prediction_list, target_list)

import os
import random

import cv2
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms.v2.functional import to_pil_image

from img2vlc.Img2VlcDataset import Img2VlcDataset
from configs import DATA_PATH, JSON_PATH
from utils import visualize, DEVICE, get_tok2vlc
from img2vlc.img2vlc_configs import IMG2VLC_VERSION, COORD_CONV_2D_VERSION
from img2vlc.img2vlc_utils import get_model, collate_fn, load_model, load_checkpoint

model = load_model(get_model(IMG2VLC_VERSION, COORD_CONV_2D_VERSION, DEVICE), load_checkpoint(DEVICE))
model.eval()

test_data_count = 10

json_list = random.sample([file
                           for file in os.listdir(os.path.join(DATA_PATH, JSON_PATH))
                           if file.endswith(".json")], test_data_count)

dataset = Img2VlcDataset(DATA_PATH, json_list, DEVICE, False)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

tok2vlc = get_tok2vlc()

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
            'vlcs': [tok2vlc[label]
                     for label in prediction['labels'].cpu().detach().numpy()]
        })
        image_list.append(cv2.cvtColor(np.array(to_pil_image(image)), cv2.COLOR_RGB2BGR))
        target_list.append({
            'boxes': target['boxes'].cpu().detach().numpy(),
            'vlcs': [tok2vlc[label]
                     for label in target['labels'].cpu().detach().numpy()]
        })

visualize(image_list, prediction_list, target_list)

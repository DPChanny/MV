import os
import random

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from MVDataset import MVDataset
from utils import get_visible_latex_char_map, DATA_PATH, JSON_PATH, get_model, visualize

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


model = get_model(device)
model.load_state_dict(torch.load(".\\check_point.pth", map_location=device)['state_dict'])
model.eval()

json_list = random.sample([file
                           for file in os.listdir(os.path.join(DATA_PATH, JSON_PATH))
                           if file.endswith(".json")][:2000], 10)

dataset = MVDataset(DATA_PATH, json_list, device, False)
dataloader = DataLoader(dataset, shuffle=False)

visible_latex_char_map = get_visible_latex_char_map()
visible_latex_char_map = {v: k for k, v in visible_latex_char_map.items()}

prediction_list = []
image_list = []

images: Tensor
for test_data_index, (images, targets) in enumerate(dataloader):
    print("PROGRESS {:.2f}%".format(test_data_index / len(dataloader) * 100))

    tensor_prediction_list = model(images)

    for prediction_index, prediction in enumerate(tensor_prediction_list):
        boxes = prediction['boxes'].detach().numpy()
        labels = prediction['labels'].detach().numpy()
        scores = prediction['scores'].detach().numpy()

        visible_latex_chars = [visible_latex_char_map[label]
                               for label_index, label in enumerate(labels)]

        prediction_list.append({
            'boxes': boxes,
            'labels': labels,
            'scores': scores,
            'visible_latex_chars': visible_latex_chars
        })

        image_list.append(images[prediction_index])

visualize(image_list, prediction_list)

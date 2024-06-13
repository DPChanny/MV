import os
import random

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from MVDataset import MVDataset
from utils import get_visible_latex_char_map, DATA_PATH, JSON_PATH

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                                trainable_backbone_layers=5,
                                weights_backbone=ResNet50_Weights.DEFAULT)

num_classes = len(get_visible_latex_char_map()) + 1
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load(".\\epoch1.pth", map_location=device))

model.eval()

json_list = random.sample([file
                           for file in os.listdir(os.path.join(DATA_PATH, JSON_PATH))
                           if file.endswith(".json")], 10)


dataset = MVDataset(DATA_PATH, json_list, device, False)
dataloader = DataLoader(dataset, shuffle=False)

for test_data_index, (image, targets) in enumerate(dataloader):
    targets[0]['boxes'].squeeze_(0)
    targets[0]['labels'].squeeze_(0)

    predictions = model(image)

    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    visible_latex_char_map = get_visible_latex_char_map()
    visible_latex_char_map = {v: k for k, v in visible_latex_char_map.items()}
    visible_latex_chars = [visible_latex_char_map[label] for label in labels.tolist()]
    scores = predictions[0]['scores']
    print(image.shape)
    print(boxes)
    print(labels)
    print(visible_latex_chars)
    print(scores)

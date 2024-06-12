import os.path
import time

import torch
import torchvision
import torchvision.transforms.v2 as tt
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from MVDataset import MVDataset
from utils import get_visible_latex_char_map, DATA_PATH

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                                trainable_backbone_layers=5,
                                weights_backbone=ResNet50_Weights.DEFAULT)

num_classes = len(get_visible_latex_char_map()) + 1
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


def get_transform(train):
    transforms = [tt.PILToTensor(), tt.ToDtype(torch.float32, scale=True), tt.Resize(size=600, max_size=1000)]
    if train:
        transforms.append(tt.RandomHorizontalFlip(0.5))
    return tt.Compose(transforms)


EPOCHS = 10
LR = 1e-5
ETA_MIN = 1e-6

optimizer = torch.optim.SGD(params=model.parameters(), lr=LR, momentum=0.95, weight_decay=1e-5 * 5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=ETA_MIN)

if os.path.exists(".\\check_point.pth"):
    check_point = torch.load(".\\check_point.pth")
    last_epoch = check_point['epoch']
    last_batch_data_index = check_point['iter']
    model.load_state_dict(check_point['state_dict'])
    optimizer.load_state_dict(check_point['optimizer'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=ETA_MIN,
                                                           last_epoch=last_epoch)
    scheduler.load_state_dict(check_point['scheduler'])
else:
    last_epoch = 0
    last_batch_data_index = 0

print("last_epoch = {} , last_batch_data_index = {}".format(last_epoch, last_batch_data_index))

epoch_term = 10
batch_term = 5

model.train()
start = time.time()

for epoch in range(last_epoch, EPOCHS):
    dataset = MVDataset(DATA_PATH, device, get_transform(train=True))
    dataloader = DataLoader(dataset)

    for batch_data_index, (image, targets) in enumerate(dataloader, last_batch_data_index):
        optimizer.zero_grad()

        targets[0]['boxes'].squeeze_(0)
        targets[0]['labels'].squeeze_(0)

        loss = model(image, targets)

        loss_objectness = loss['loss_objectness']
        loss_rpn_box_reg = loss['loss_rpn_box_reg']
        loss_classifier = loss['loss_classifier']
        loss_box_reg = loss['loss_box_reg']

        rpn_total = loss_objectness + 10 * loss_rpn_box_reg
        fast_rcnn_total = loss_classifier + 1 * loss_box_reg

        total_loss = rpn_total + fast_rcnn_total

        if batch_data_index % batch_term == 0:
            end = time.time()
            print(("epoch: {} batch_data_index: {} "
                   + "| loss_objectness: {:.5f} loss_rpn_box_reg: {:.5f} "
                   + "loss_classifier: {:.5f} loss_box_reg: {:.5f} "
                   + "| Duration: {:.5f}").format(epoch,
                                                  batch_data_index,
                                                  loss['loss_objectness'],
                                                  loss['loss_rpn_box_reg'],
                                                  loss['loss_classifier'],
                                                  loss['loss_box_reg'],
                                                  end - start))

            torch.save({'epoch': epoch,
                        'iter': batch_data_index,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                        }, "check_point.pth")

            start = time.time()

        total_loss.backward()
        optimizer.step()

    last_batch_data_index = 0
    scheduler.step()

    if epoch % epoch_term == 0:
        torch.save(model.state_dict(), "epoch{}.pth".format(epoch))

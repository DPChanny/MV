import os.path
import time

import torch
from torch.utils.data import DataLoader

from MVDataset import MVDataset
from utils import DATA_PATH, JSON_PATH, collate_fn, get_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = get_model(device)

EPOCHS = 10
LR = 1e-5
ETA_MIN = LR * 1e-1

optimizer = torch.optim.SGD(params=model.parameters(), lr=LR, momentum=0.95, weight_decay=1e-5 * 5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=ETA_MIN)

if os.path.exists(".\\check_point.pth"):
    check_point = torch.load(".\\check_point.pth", map_location=device)
    start_epoch = check_point['epoch']
    start_batch = check_point['batch'] + 1
    model.load_state_dict(check_point['state_dict'])
    optimizer.load_state_dict(check_point['optimizer'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS,
                                                           last_epoch=start_epoch, eta_min=ETA_MIN)
    scheduler.load_state_dict(check_point['scheduler'])
else:
    start_epoch = 0
    start_batch = 0

EPOCH_TERM = 1
BATCH_SIZE = 100
MINI_BATCH_SIZE = 4
MINI_BATCH_VERBOSE_TERM = 5

model.train()

json_list = [file for file in os.listdir(os.path.join(DATA_PATH, JSON_PATH)) if file.endswith(".json")]
batch_json_lists = []
for index in range(0, len(json_list), BATCH_SIZE):
    batch_json_lists.append(json_list[index:min(index + BATCH_SIZE, len(json_list))])

for epoch in range(start_epoch, EPOCHS):
    for batch, batch_json_list in enumerate(batch_json_lists, start_batch):
        start = time.time()

        dataset = MVDataset(DATA_PATH, batch_json_list, device, True)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=4, collate_fn=collate_fn)

        loss_history_list = []
        for mini_batch_data_index, (images, targets) in enumerate(dataloader):
            optimizer.zero_grad()

            loss = model(images, targets)
            total_loss = torch.sum(torch.stack([value for value in loss.values()]))

            loss_history = {key: value.item() for (key, value) in loss.items()}
            loss_history['total_loss'] = total_loss.item()
            loss_history_list.append(loss_history)

            if not mini_batch_data_index % MINI_BATCH_VERBOSE_TERM:
                end = time.time()

                loss_history_sum = {}
                for loss_history in loss_history_list:
                    for (key, value) in loss_history.items():
                        if loss_history_sum.get(key) is None:
                            loss_history_sum[key] = 0
                        loss_history_sum[key] += value

                loss_history_mean = {}
                for (key, value) in loss_history_sum.items():
                    if loss_history_mean.get(key) is None:
                        loss_history_mean[key] = 0
                    loss_history_mean[key] = value / MINI_BATCH_VERBOSE_TERM

                loss_history_list.clear()

                print(("EPOCH {}/{} | BATCH {}/{} | PROGRESS {:.2f}% "
                       + "| MEAN TOTAL LOSS {:.5f}"
                       + "| MEAN LOSS [objectness: {:.5f}, rpn_box_reg: {:.5f}, "
                       + "classifier: {:.5f}, box_reg: {:.5f}] "
                       + "| TIME: {:.5f}").format(epoch, EPOCHS,
                                                  batch, len(batch_json_lists),
                                                  mini_batch_data_index / len(dataloader) * 100,
                                                  loss_history_mean['total_loss'],
                                                  loss_history_mean['loss_objectness'],
                                                  loss_history_mean['loss_rpn_box_reg'],
                                                  loss_history_mean['loss_classifier'],
                                                  loss_history_mean['loss_box_reg'],
                                                  end - start))

                start = time.time()

            total_loss.backward()
            optimizer.step()

        torch.save({'epoch': epoch,
                    'batch': batch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                    }, "check_point.pth")
        scheduler.step()

    if not epoch % EPOCH_TERM:
        torch.save(model.state_dict(), "epoch{}.pth".format(epoch))

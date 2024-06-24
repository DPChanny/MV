import os.path
import time

import torch
from torch.utils.data import DataLoader

from MVDataset import MVDataset
from configs import DATA_PATH, JSON_PATH
from utils import Timer, DEVICE
from vlc_detector_configs import VLC_DETECTOR_VERSION, COORD_CONV_2D_VERSION
from vlc_detector_utils import (collate_fn, get_model, save_checkpoint,
                                load_checkpoint, load_optimizer, load_starts, load_model, load_scheduler)

EPOCHS = 10
LEARNING_RATE = 1e-3
ETA_MIN = 1e-6
VERBOSE = 50
BATCH_SIZE = 2000
MINI_BATCH_SIZE = 2
WEIGHT_DECAY = 0.0005

checkpoint = load_checkpoint(DEVICE)

model = load_model(get_model(VLC_DETECTOR_VERSION, COORD_CONV_2D_VERSION, DEVICE), checkpoint)
model.train()

optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
optimizer = load_optimizer(optimizer, checkpoint)

start_epoch, start_batch = load_starts(checkpoint)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=ETA_MIN)
scheduler = load_scheduler(scheduler, checkpoint)

json_list = [file for file in os.listdir(os.path.join(DATA_PATH, JSON_PATH)) if file.endswith(".json")]
batch_json_lists = []
for index in range(0, len(json_list), BATCH_SIZE):
    batch_json_lists.append(json_list[index:min(index + BATCH_SIZE, len(json_list))])

timer = Timer()


def calc_log_history(history, title):
    mean_total = 0
    print("MEAN {} ".format(title), end='')
    for name, value_list in history.items():
        mean = sum(value_list) / len(value_list) if value_list else 0
        mean_total += mean
        print("({}: {:.5f}) ".format(name, mean), end='')
        value_list.clear()
    print("| MEAN {} TOTAL {:.5f}".format(title, mean_total))

    return mean_total


def calc_et(total_et, progress, total):
    return time.strftime("%Y %m %d %H:%M:%S",
                         time.localtime(time.time() + total_et * (total - progress - 1) / total))


loss_history = {"loss_objectness": list(), "loss_rpn_box_reg": list(),
                "loss_classifier": list(), "loss_box_reg": list()}

duration_history = {'load': list(), 'zero_grad': list(),
                    'forward': list(), 'loss_history': list(),
                    'backward': list(), 'step': list()}

for epoch in range(start_epoch, EPOCHS):
    for batch in range(start_batch, len(batch_json_lists)):
        save_checkpoint(epoch, EPOCHS, batch, len(batch_json_lists),
                        model, optimizer, scheduler)

        dataset = MVDataset(DATA_PATH, batch_json_lists[batch], DEVICE, True)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=MINI_BATCH_SIZE, collate_fn=collate_fn)

        timer.start()
        for mini_batch_data_index, (images, targets) in enumerate(dataloader):
            duration_history['load'].append(timer.end())

            optimizer.zero_grad()
            duration_history['zero_grad'].append(timer.end())

            loss = model(images, targets)
            duration_history['forward'].append(timer.end())

            for key, value in loss.items():
                loss_history[key].append(value.item())
            duration_history['loss_history'].append(timer.end())

            total_loss = torch.sum(torch.stack([value for value in loss.values()]))
            total_loss.backward()
            duration_history['backward'].append(timer.end())

            optimizer.step()
            duration_history['step'].append(timer.end())

            print("EPOCH {}/{} | BATCH {}/{} | PROGRESS {}/{}".format(epoch, EPOCHS, batch, len(batch_json_lists),
                                                                      mini_batch_data_index, len(dataloader)))

            if not mini_batch_data_index % VERBOSE:
                calc_log_history(loss_history, "LOSS")
                mean_duration_total = calc_log_history(duration_history, "DURATION")

                et_batch = mean_duration_total * len(dataloader)
                et_epoch = et_batch * len(batch_json_lists)
                et_train = et_epoch * EPOCHS

                print("ESTIMATED TIME (train: {}) (epoch: {}) (batch: {})".format(
                    calc_et(et_train, epoch, EPOCHS),
                    calc_et(et_epoch, batch, len(batch_json_lists)),
                    calc_et(et_batch, mini_batch_data_index, len(dataloader))))

            timer.start()

        del dataset, dataloader

    start_batch = 0

    scheduler.step()

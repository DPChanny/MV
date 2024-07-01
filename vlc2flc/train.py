import os.path
import time

import torch
from torch.utils.data import DataLoader

from configs import DATA_PATH, JSON_PATH, DEVICE
from utils import Timer, PAD_INDEX
from vlc2flc.Vlc2FlcDataset import Vlc2FlcDataset
from vlc2flc.vlc2flc_utils import (collate_fn, create_mask, load_checkpoint, save_checkpoint,
                                   get_model, load_model, load_optimizer, load_starts, load_scheduler)

EPOCHS = 100
LEARNING_RATE = 1e-3
ETA_MIN = 1e-6
VERBOSE = 50
BATCH_SIZE = 2000
MINI_BATCH_SIZE = 20

checkpoint = load_checkpoint(DEVICE)

model = load_model(get_model(DEVICE), checkpoint)
model.train()

optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE, weight_decay=1e-03)
optimizer = load_optimizer(optimizer, checkpoint)

start_epoch, start_batch = load_starts(checkpoint)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS - 1,
                                                       last_epoch=start_epoch - 1, eta_min=ETA_MIN)
scheduler = load_scheduler(scheduler, checkpoint)

json_list = [file for file in os.listdir(os.path.join(DATA_PATH, JSON_PATH)) if file.endswith(".json")]
batch_json_lists = []
for index in range(0, len(json_list), BATCH_SIZE):
    batch_json_lists.append(json_list[index:min(index + BATCH_SIZE, len(json_list))])

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_INDEX)

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


loss_history = {'loss': list()}

duration_history = {'load': list(), 'zero_grad': list(),
                    'forward': list(), 'loss_history': list(),
                    'backward': list(), 'step': list()}

for epoch in range(start_epoch, EPOCHS):
    print("LEARNING RATE {:.6f}".format(optimizer.param_groups[0]['lr']))
    for batch in range(start_batch, len(batch_json_lists)):
        save_checkpoint(epoch, EPOCHS, batch, len(batch_json_lists),
                        model, optimizer, scheduler)

        dataset = Vlc2FlcDataset(DATA_PATH, batch_json_lists[batch], DEVICE)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=MINI_BATCH_SIZE, collate_fn=collate_fn)

        timer.start()
        for mini_batch_data_index, (src, tgt, boxes) in enumerate(dataloader):
            duration_history['load'].append(timer.end())

            optimizer.zero_grad()
            duration_history['zero_grad'].append(timer.end())

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device=DEVICE)

            logits = model(src, boxes, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
            duration_history['forward'].append(timer.end())

            tgt_output = tgt[1:, :]

            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
            loss_history['loss'].append(loss.item())
            duration_history['loss_history'].append(timer.end())

            loss.backward()
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

save_checkpoint(EPOCHS, EPOCHS, len(batch_json_lists), len(batch_json_lists),
                model, optimizer, scheduler)

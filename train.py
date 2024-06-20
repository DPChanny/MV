import os.path
import time

import torch
from torch.utils.data import DataLoader

from MVDataset import MVDataset
from config import MODEL_VERSION, COORD_CONV_2D_VERSION, MODEL_PATH, DATA_PATH, JSON_PATH
from utils import collate_fn, get_model, Timer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = get_model(MODEL_VERSION, COORD_CONV_2D_VERSION, device)

EPOCHS = 10
LEARNING_RATE = 1e-3
ETA_MIN = 1e-6
VERBOSE = 50
BATCH_SIZE = 2000
MINI_BATCH_SIZE = 2

optimizer = torch.optim.AdamW(params=model.parameters(),
                              lr=LEARNING_RATE,
                              weight_decay=0.0005)

if os.path.exists(os.path.join(MODEL_PATH,
                               str(MODEL_VERSION),
                               str(COORD_CONV_2D_VERSION),
                               "check_point.pth")):
    check_point = torch.load(os.path.join(MODEL_PATH,
                                          str(MODEL_VERSION),
                                          str(COORD_CONV_2D_VERSION),
                                          "check_point.pth"),
                             map_location=device)
    start_epoch = check_point['start_epoch']
    start_batch = check_point['start_batch']
    model.load_state_dict(check_point['model'])
    optimizer.load_state_dict(check_point['optimizer'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           EPOCHS,
                                                           last_epoch=start_epoch,
                                                           eta_min=ETA_MIN)
    scheduler.load_state_dict(check_point['scheduler'])
else:
    start_epoch = 0
    start_batch = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           EPOCHS,
                                                           eta_min=ETA_MIN)

model.train()

json_list = [file
             for file in os.listdir(os.path.join(DATA_PATH, JSON_PATH))
             if file.endswith(".json")]
batch_json_lists = []
for index in range(0, len(json_list), BATCH_SIZE):
    batch_json_lists.append(json_list[index:min(index + BATCH_SIZE, len(json_list))])

if not os.path.exists(os.path.join(MODEL_PATH,
                                   str(MODEL_VERSION),
                                   str(COORD_CONV_2D_VERSION))):
    os.makedirs(os.path.join(MODEL_PATH,
                             str(MODEL_VERSION),
                             str(COORD_CONV_2D_VERSION)))

timer = Timer()

for epoch in range(start_epoch, EPOCHS):
    for batch in range(start_batch, len(batch_json_lists)):
        values = {'start_epoch': epoch,
                  'start_batch': batch,
                  'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict()}

        torch.save(values, os.path.join(MODEL_PATH,
                                        str(MODEL_VERSION),
                                        str(COORD_CONV_2D_VERSION),
                                        "{}-{}_{}-{}.pth".format(epoch, EPOCHS,
                                                                 batch, len(batch_json_lists))))
        torch.save(values, os.path.join(MODEL_PATH,
                                        str(MODEL_VERSION),
                                        str(COORD_CONV_2D_VERSION),
                                        "check_point.pth".format(epoch, EPOCHS,
                                                                 batch, len(batch_json_lists))))

        dataset = MVDataset(DATA_PATH, batch_json_lists[batch], device, True)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=MINI_BATCH_SIZE, collate_fn=collate_fn)

        loss_history = {"loss_objectness": list(), "loss_rpn_box_reg": list(),
                        "loss_classifier": list(), "loss_box_reg": list()}

        duration_history = {'forward': list(), 'backward': list(),
                            'loss_history': list(), 'load': list(),
                            'zero_grad': list(), 'step': list()}

        timer.start()
        for mini_batch_data_index, (images, targets) in enumerate(dataloader):
            duration_history['load'].append(timer.end())

            optimizer.zero_grad()
            duration_history['zero_grad'].append(timer.end())

            timer.start()
            loss = model(images, targets)
            total_loss = torch.sum(torch.stack([value for value in loss.values()]))
            duration_history['forward'].append(timer.end())

            for key, value in loss.items():
                loss_history[key].append(value.item())
            duration_history['loss_history'].append(timer.end())

            total_loss.backward()
            duration_history['backward'].append(timer.end())

            optimizer.step()
            duration_history['step'].append(timer.end())

            print("EPOCH {}/{} | BATCH {}/{} | PROGRESS {}/{}".format(epoch, EPOCHS,
                                                                      batch, len(batch_json_lists),
                                                                      mini_batch_data_index, len(dataloader)))

            if not mini_batch_data_index % VERBOSE:
                print("MEAN LOSS ", end='')
                mean_loss_total = 0
                for name, value in loss_history.items():
                    mean_loss_total += sum(value) / len(value)
                    print("({}: {:.5f}) ".format(name, sum(value) / len(value)), end='')
                    value.clear()
                print("| MEAN LOSS TOTAL {:.5f}".format(mean_loss_total))

                mean_duration_total = 0
                print("MEAN DURATION ", end='')
                for name, value in duration_history.items():
                    mean_duration_total += sum(value) / len(value)
                    print("({}: {:.3f}) ".format(name, sum(value) / len(value)), end='')
                    value.clear()
                print("| MEAN DURATION TOTAL {:.3f} ".format(mean_duration_total))

                et_batch = mean_duration_total * len(dataloader)
                et_epoch = et_batch * len(batch_json_lists)
                et_train = et_epoch * EPOCHS

                def get_time_format(seconds):
                    return time.strftime("%Y %m %d %H:%M:%S", time.localtime(seconds))

                def get_ratio(current, total):
                    return (total - current - 1) / total


                print("ESTIMATED TIME (train: {}) (epoch: {}) (batch: {})".format(
                    get_time_format(time.time() + et_train * get_ratio(epoch, EPOCHS)),
                    get_time_format(time.time() + et_epoch * get_ratio(batch, len(batch_json_lists))),
                    get_time_format(time.time() + et_batch * get_ratio(mini_batch_data_index, len(dataloader)))))

            timer.start()

        del dataset, dataloader

    scheduler.step()

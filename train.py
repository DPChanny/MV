import os.path
import time

import torch
from torch.utils.data import DataLoader

from MVDataset import MVDataset
from config import MODEL_VERSION, COORD_CONV_2D_VERSION, MODEL_PATH, DATA_PATH, JSON_PATH
from utils import collate_fn, get_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = get_model(MODEL_VERSION, COORD_CONV_2D_VERSION, device)

EPOCHS = 5
LEARNING_RATE = 1e-3
VERBOSE_TERM = 1
BATCH_SIZE = 1000
MINI_BATCH_SIZE = 2

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS,
                                                           last_epoch=start_epoch)
    scheduler.load_state_dict(check_point['scheduler'])
else:
    start_epoch = 0
    start_batch = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

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

for epoch in range(start_epoch, EPOCHS):
    for batch, batch_json_list in enumerate(batch_json_lists, start_batch):
        torch.save({'start_epoch': epoch,
                    'start_batch': batch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                    }, os.path.join(MODEL_PATH,
                                    str(MODEL_VERSION),
                                    str(COORD_CONV_2D_VERSION),
                                    "{}-{}_{}-{}.pth".format(epoch, EPOCHS,
                                                             batch, BATCH_SIZE)))

        dataset = MVDataset(DATA_PATH, batch_json_list, device, True)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=MINI_BATCH_SIZE, collate_fn=collate_fn)

        start = time.time()

        for mini_batch_data_index, (images, targets) in enumerate(dataloader):
            optimizer.zero_grad()

            loss = model(images, targets)
            total_loss = torch.sum(torch.stack([value for value in loss.values()]))

            if not mini_batch_data_index % VERBOSE_TERM:
                end = time.time()

                print(("EPOCH {}/{} | BATCH {}/{} | PROGRESS {}/{} "
                       + "| TOTAL LOSS {:.3f} "
                       + "| LOSS [objectness: {:.3f}, rpn_box_reg: {:.3f}, "
                       + "classifier: {:.3f}, box_reg: {:.3f}] "
                       + "| DURATION: {:.2f}s").format(epoch, EPOCHS,
                                                       batch, len(batch_json_lists),
                                                       mini_batch_data_index, len(dataloader),
                                                       total_loss,
                                                       loss['loss_objectness'], loss['loss_rpn_box_reg'],
                                                       loss['loss_classifier'], loss['loss_box_reg'],
                                                       end - start))

                start = time.time()

            total_loss.backward()
            optimizer.step()

    scheduler.step()

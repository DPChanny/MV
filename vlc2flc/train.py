import os.path
import time

import torch
from torch.utils.data import DataLoader

from img2vlc.Img2VlcDataset import Img2VlcDataset
from configs import DATA_PATH, JSON_PATH
from utils import Timer, DEVICE, get_vlc2tok, get_flc2tok
from vlc2flc.Vlc2FlcDataset import Vlc2FlcDataset
from vlc2flc.vlc2flc_utils import Seq2SeqTransformer, collate_fn, create_mask

EPOCHS = 100
EMB_SIZE = 128
LEARNING_RATE = 1e-3
ETA_MIN = 1e-6
VERBOSE = 50
BATCH_SIZE = 2000
MINI_BATCH_SIZE = 2
WEIGHT_DECAY = 0.0005

model = Seq2SeqTransformer(EMB_SIZE, 4,
                           10000, 10000,
                           6, 6,
                           len(get_vlc2tok()), len(get_flc2tok()), device=DEVICE)
model.train()

optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS - 1, eta_min=ETA_MIN)

json_list = [file for file in os.listdir(os.path.join(DATA_PATH, JSON_PATH)) if file.endswith(".json")]
batch_json_lists = []
for index in range(0, len(json_list), BATCH_SIZE):
    batch_json_lists.append(json_list[index:min(index + BATCH_SIZE, len(json_list))])

model.to(DEVICE)

for epoch in range(EPOCHS):
    print("LEARNING RATE {:.6f}".format(optimizer.param_groups[0]['lr']))
    for batch in range(len(batch_json_lists)):
        dataset = Vlc2FlcDataset(DATA_PATH, batch_json_lists[batch], DEVICE)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=MINI_BATCH_SIZE, collate_fn=collate_fn)

        for mini_batch_data_index, (src, tgt, boxes) in enumerate(dataloader):
            optimizer.zero_grad()

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt, device=DEVICE)

            loss = model(src, boxes, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)

            print(loss)

            total_loss = torch.sum(torch.stack([value for value in loss.values()]))
            total_loss.backward()

            optimizer.step()

            print("EPOCH {}/{} | BATCH {}/{} | PROGRESS {}/{}".format(epoch, EPOCHS, batch, len(batch_json_lists),
                                                                      mini_batch_data_index, len(dataloader)))

        del dataset, dataloader

    start_batch = 0

    scheduler.step()

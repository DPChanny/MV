import os
import random

import matplotlib
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from configs import DATA_PATH, JSON_PATH, DEVICE
from utils import get_tok2flc
from vlc2flc.Vlc2FlcDataset import Vlc2FlcDataset
from vlc2flc.vlc2flc_utils import get_model, collate_fn, load_model, load_checkpoint

matplotlib.rcParams["mathtext.fontset"] = "cm"


def show(t1, t2):
    fig = plt.figure(figsize=(10, 2))
    fig.text(
        x=0.3333,
        y=0.5,
        s="$" + t1 + "$",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=16,
    )

    fig.text(
        x=0.6666,
        y=0.5,
        s="$" + t2 + "$",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=16,
    )

    plt.show()


checkpoint = load_checkpoint(DEVICE)

model = load_model(get_model(DEVICE), checkpoint)

model.eval()

test_data_count = 10

json_list = random.sample([file
                           for file in os.listdir(os.path.join(DATA_PATH, JSON_PATH))
                           if file.endswith(".json")], test_data_count)

dataset = Vlc2FlcDataset(DATA_PATH, json_list, DEVICE)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

tok2flc = get_tok2flc()

for test_data_index, (src, tgt, boxes) in enumerate(dataloader):
    result = model.translate(src, boxes)

    src = [tok2flc[tok.item()] for tok in src]
    tgt = "".join([tok2flc[tok.item()] for tok in tgt[1:-1, :]])
    result = "".join([tok2flc[tok.item()] for tok in result[1:-1, :]])

    print(src)
    show(tgt, result)

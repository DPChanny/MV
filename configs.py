import os

import torch

PROJECT_PATH: str = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_PATH: str = os.path.join(PROJECT_PATH, "raw_data")
DATA_PATH: str = os.path.join(PROJECT_PATH, "data")

JSON_PATH: str = "jsons"
JPG_PATH: str = "jpgs"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

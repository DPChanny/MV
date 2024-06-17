import os
from enum import Enum


class ModelVersion(Enum):
    V1_PRETRAINED = 0
    V2_PRETRAINED = 1
    V1 = 2
    V2 = 3


class CoordConv2dVersion(Enum):
    NONE = 0
    V1 = 1


MODEL_VERSION = ModelVersion.V2_PRETRAINED
COORD_CONV_2D_VERSION = CoordConv2dVersion.V1

PROJECT_PATH: str = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH: str = os.path.join(PROJECT_PATH, "models")

RAW_DATA_PATH: str = os.path.join(PROJECT_PATH, "raw_data")
DATA_PATH: str = os.path.join(PROJECT_PATH, "data")

JSON_PATH: str = "jsons"
JPG_PATH: str = "jpgs"

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


class OptimizerVersion(Enum):
    SGD = 0,
    ADAM = 1,
    ADAM_W = 2


IMG2VLC_VERSION = ModelVersion.V2_PRETRAINED
COORD_CONV_2D_VERSION = CoordConv2dVersion.V1
OPTIMIZER_VERSION = OptimizerVersion.ADAM_W

MODEL_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

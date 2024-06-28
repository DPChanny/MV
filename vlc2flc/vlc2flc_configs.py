import os

from utils import get_flc2tok

PAD_INDEX = len(get_flc2tok()) + 1
SOS_INDEX = len(get_flc2tok()) + 2
EOS_INDEX = len(get_flc2tok()) + 3

MODEL_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

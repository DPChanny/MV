import torchvision
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

from utils import get_visible_latex_char_map, get_model, ModelVersion

model = get_model(ModelVersion.V2, "cpu")

print(model)

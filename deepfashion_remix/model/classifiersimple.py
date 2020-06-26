import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

class clssimp(nn.Module):
    def __init__(self, ch=2048, num_classes=20):

        super(clssimp, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.cls= nn.Linear(ch, num_classes,bias=True)

    def forward(self, x):
        # bp()
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        logits = self.cls(x)

        return logits



import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision

class Identity(nn.Module):
    def __int__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
model = torchvision.models.vgg16(pretrained=True)
model.avgpool[0] = Identity()
model.classifier = nn.Sequential(nn.Linear(512, 10),
                                nn.ReLU(),
                                nn.Linear(100, 10))
model.to("cpu") #used to cotinue model training process



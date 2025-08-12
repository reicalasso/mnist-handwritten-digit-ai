import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18_1ch(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super().__init__()
        # torchvision'un resnet18'ini al, conv1'i 1 kanala uyarlayalım
        self.model = resnet18(pretrained=pretrained)
        # orijinal conv1: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Bunu 1 kanala çek
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # son fully-connected katmanı yeniden boyutlandır
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
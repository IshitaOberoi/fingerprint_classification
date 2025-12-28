import torch.nn as nn
from torchvision import models


def get_resnet18(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_mobilenet_v2(num_classes):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, num_classes
    )
    return model

import torch
from torch import nn
from torchvision.models import resnet101


class KaggleModel(nn.Module):

    def __init__(self, n_classes):
        """
        out_units: trainable param for model output logits
        """
        super(KaggleModel, self).__init__()
        resnet = resnet101(pretrained=True)
        for param in resnet.parameters(): # Freeze weights
            param.requires_grad = False
        self.resnet = nn.Sequential(*list(resnet.children())[:-1],
                                     nn.Flatten(),
                                     nn.Linear(resnet.fc.in_features, n_classes))

    def forward(self, x):
        return self.resnet(x)

    def __call__(self, x):
        return self.forward(x)


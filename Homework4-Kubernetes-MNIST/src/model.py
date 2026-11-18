"""CNN and preprocessing adapted from the Homework 3 PyTorch MNIST example.

Original: https://github.com/pytorch/examples/tree/main/mnist
Copyright (c) 2017, Pytorch contributors. BSD 3-Clause; see LICENSE.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, images):
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return F.log_softmax(self.fc2(x), dim=1)


# Training and inference must apply the same normalization.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

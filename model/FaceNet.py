import math

import torch
from torch import nn
from torch.nn import functional as F


def init_weights_snn(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(std=math.sqrt(1 / m.in_features))
    elif isinstance(m, nn.Conv2d):
        m.weight.data.normal_(std=math.sqrt(1 / (m.in_channels * m.kernel_size[0] * m.kernel_size[1])))
    if hasattr(m, 'bias') and hasattr(m.bias, 'data'):
        m.bias.data.zero_()


class FaceNet(nn.Module):
    def __init__(self, r_dim, in_c=3):
        super(FaceNet, self).__init__()

        self.conv1 = nn.Conv2d(in_c, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2a = nn.Conv2d(64, 64, 1, 1)
        self.conv2 = nn.Conv2d(64, 192, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(192)

        self.conv3a = nn.Conv2d(192, 192, 1, 1)
        self.conv3 = nn.Conv2d(192, 384, 3, 1, 1)

        self.conv4a = nn.Conv2d(384, 384, 1, 1)
        self.conv4 = nn.Conv2d(384, 256, 3, 1, 1)

        self.conv5a = nn.Conv2d(256, 256, 1, 1)
        self.conv5 = nn.Conv2d(256, 256, 3, 1, 1)

        self.conv6a = nn.Conv2d(256, 256, 1, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)

        self.fc1 = nn.Linear(4096, 128)
        self.fc2 = nn.Linear(128, r_dim)

        self.apply(init_weights_snn)

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        x = self.bn1(x)
        x = F.selu(self.conv2a(x))
        x = F.selu(self.conv2(x))
        x = self.bn2(x)
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        x = F.selu(self.conv3a(x))
        x = F.selu(self.conv3(x))
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        x = F.selu(self.conv4a(x))
        x = F.selu(self.conv4(x))
        x = F.selu(self.conv5a(x))
        x = F.selu(self.conv5(x))
        x = F.selu(self.conv6a(x))
        x = F.selu(self.conv6(x))
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.tanh(x)

        return x

    def encode(self, x):
        return self.forward(x)

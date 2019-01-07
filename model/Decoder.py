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


class ConvDecLayer(nn.Module):
    def __init__(self, c_in, c_out=None, final=False):
        super(ConvDecLayer, self).__init__()
        if c_out is None:
            c_out = c_in // 2
        self.net = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, 4, 2, 1, bias=True),
        )
        if not final:
            self.net.add_module('activ', nn.SELU(inplace=True))

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, shape, r_dim, out_c):
        super(Decoder, self).__init__()
        h_dim = 2048

        self.NUM_LAYERS = 6
        self.C_END = 32

        self.c_start = self.C_END * (2**(self.NUM_LAYERS - 1))
        self.shape_start = (shape[0] // 2**self.NUM_LAYERS, shape[1] // 2**self.NUM_LAYERS)
        self.h0_dim = self.c_start * self.shape_start[0] * self.shape_start[1]

        self.fc = nn.Sequential(
            nn.Linear(r_dim, h_dim),
            nn.SELU(),
            nn.Linear(h_dim, h_dim),
            nn.SELU(),
            nn.Linear(h_dim, self.h0_dim),
            nn.SELU(),
        )

        self.conv = nn.Sequential()
        for i in range(self.NUM_LAYERS - 1, 0, -1):
            self.conv.add_module(f'conv_{i}', ConvDecLayer(self.C_END * (2**i)))
        self.conv.add_module(f'conv_0', ConvDecLayer(c_in=self.C_END, c_out=32, final=True))

        self.conv2 = nn.Sequential(
            nn.SELU(),
            nn.Conv2d(32, 128, 1, 1, 0),
            nn.SELU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.SELU(),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.SELU(),
            nn.Conv2d(64, out_c, 7, 1, 3),
        )

        self.apply(init_weights_snn)

    def forward(self, z):
        y = self.fc(z).view(-1, self.c_start, *self.shape_start)
        y = self.conv(y)
        y = F.tanh(self.conv2(y))
        return y

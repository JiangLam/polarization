import torch
from torch import nn


net = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), nn.ReLU(),
    nn.AvgPool2d(kernel_size=3, stride=1),
    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), nn.ReLU(),
    nn.AvgPool2d(kernel_size=3, stride=1),
    nn.Flatten(),
    nn.Linear()
)


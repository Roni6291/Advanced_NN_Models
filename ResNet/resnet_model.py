# -*- coding: utf-8 -*-
"""
From scratch implementation of the famous ResNet models - 50, 101, 152.
The intuition for ResNet is simple and clear, but to code
it didn't feel super clear at first, even when reading Pytorch own
implementation. 

Coded by Roni Abraham on August 1, 2020>
"""


import torch
import torch.nn as nn

class Block(nn.Module):
    """
    The Repeating pattern of 3 conv layers:-

    1*1, intermediate_channels, stride=1, padding=0      |
    3*3, intermediate_channels, stride=stride, padding=1    |-- block_repeats
    1*1, intermediate_channels*4, stride=1, padding=0  |

    """
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()

        self.expansion = 4

        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)

        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)

        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)

        self.relu = nn.ReLU()

        self.identity_downsample = identity_downsample

        self.stride = stride
    
    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    """

    Resnet consists of following layers:-
    -- Initial Conv (7*7, 64, stride=2, padding=3) layer
    -- Maxpool layer (3*3, stride=2, padding=1) layer
    -- Repeating blocks as in class Block
    -- FC layers ( as desirable accourding to the number of classes)

    """

    def __init__(self, Block, layers, image_channels, num_classes):

        super(ResNet, self).__init__()

        self.in_channels = 64
        self.expansion = 4

        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # The entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)
    
    def forward(self, X_train):

        x = self.conv1(X_train)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
    
    def _make_layer(self, Block, num_blocks, intermediate_channels, stride):

        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead

        if stride != 1 or self.in_channels != intermediate_channels * self.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                ), nn.BatchNorm2d(intermediate_channels * self.expansion),)
        
        layers.append(Block(self.in_channels, intermediate_channels, identity_downsample, stride))

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * self.expansion

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_blocks - 1):
            layers.append(Block(self.in_channels, intermediate_channels))

        
        return nn.Sequential(*layers)


def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(Block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(Block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(Block, [3, 8, 36, 3], img_channel, num_classes)


# def test():
#     net = ResNet101(img_channel=3, num_classes=1000)
#     y = net(torch.randn(4, 3, 224, 224)).to("cuda")
#     print(y.size())


# test()
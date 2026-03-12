# -*- coding: utf-8 -*-
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            # 对应relu4_1，512通道
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # 对应relu3_1，256通道
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, padding=0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, padding=0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # 对应relu2_1，128通道
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3, padding=0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # 对应relu1_1，64通道
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3, padding=0),
            nn.ReLU(),

            # 输出层，还原RGB
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3, padding=0),
        )

    def forward(self, x):
        return self.decoder(x)
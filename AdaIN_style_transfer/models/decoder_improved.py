# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # ── Block1：对应relu4_1，512→256，上采样到32×32 ──
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3, padding=0),
            nn.ReLU(),
        )

        # ── concat relu3_1后融合，512→256 ──
        self.fusion3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3, padding=0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, padding=0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3, padding=0),
            nn.ReLU(),
        )

        # ── concat relu2_1后融合，256→128 ──
        self.fusion2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3, padding=0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3, padding=0),
            nn.ReLU(),
        )

        # ── concat relu1_1后融合，128→64 ──
        self.fusion1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3, padding=0),
            nn.ReLU(),
        )

        # ── 输出层，64→3 ──
        self.output = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3, padding=0),
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, t, content_feats):
        """
        t:             AdaIN输出，(B, 512, 16, 16)
        content_feats: Encoder四层特征 (relu1_1, relu2_1, relu3_1, relu4_1)
        """
        relu1_1, relu2_1, relu3_1, _ = content_feats

        # 16×16 → 32×32
        x = self.block1(t)
        x = self.upsample(x)

        # concat relu3_1：(B,256,32,32) + (B,256,32,32) = (B,512,32,32)
        x = torch.cat([x, relu3_1], dim=1)
        x = self.fusion3(x)          # → (B,128,32,32)
        x = self.upsample(x)

        # concat relu2_1：(B,128,64,64) + (B,128,64,64) = (B,256,64,64)
        x = torch.cat([x, relu2_1], dim=1)
        x = self.fusion2(x)          # → (B,64,64,64)
        x = self.upsample(x)

        # concat relu1_1：(B,64,128,128) + (B,64,128,128) = (B,128,128,128)
        x = torch.cat([x, relu1_1], dim=1)
        x = self.fusion1(x)          # → (B,64,128,128)

        # 输出层 → (B,3,128,128)
        return self.output(x)
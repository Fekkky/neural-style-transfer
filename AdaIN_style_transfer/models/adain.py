# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class AdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def calc_mean_std(self, feat):
        """
        计算特征图每个通道的均值和标准差
        feat: (B, C, H, W)
        """
        B, C, H, W = feat.shape
        feat_flat = feat.view(B, C, -1)              # (B, C, H*W)
        mean = feat_flat.mean(dim=2).view(B, C, 1, 1)
        std  = feat_flat.std(dim=2).view(B, C, 1, 1) + self.eps
        return mean, std

    def forward(self, content_feat, style_feat):
        """
        content_feat: 内容图特征 (B, C, H, W)
        style_feat:   风格图特征 (B, C, H, W)
        """
        # 计算内容图和风格图各自的均值和标准差
        c_mean, c_std = self.calc_mean_std(content_feat)
        s_mean, s_std = self.calc_mean_std(style_feat)

        # 先归一化内容图特征（去除内容图原有风格）
        normalized = (content_feat - c_mean) / c_std

        # 再用风格图的统计重新缩放（施加目标风格）
        return s_std * normalized + s_mean
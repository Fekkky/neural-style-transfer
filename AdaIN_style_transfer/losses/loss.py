# -*- coding: utf-8 -*-
import torch.nn as nn
import torch

def calc_mean_std(feat, eps=1e-5):
    B, C, H, W = feat.shape
    feat_flat = feat.view(B, C, -1)
    mean = feat_flat.mean(dim=2).view(B, C, 1, 1)
    std  = feat_flat.std(dim=2).view(B, C, 1, 1) + eps
    return mean, std

def content_loss(generated_feat, target_feat):
    """
    内容损失：AdaIN输出的特征和生成图经过Encoder后的特征比较
    """
    return nn.MSELoss()(generated_feat, target_feat)

def style_loss(generated_feats, style_feats):
    """
    风格损失：直接比较均值和标准差，不用Gram矩阵
    generated_feats：生成图在多层的特征列表
    style_feats：    风格图在多层的特征列表
    """
    loss = 0
    for gen_feat, style_feat in zip(generated_feats, style_feats):
        gen_mean,   gen_std   = calc_mean_std(gen_feat)
        style_mean, style_std = calc_mean_std(style_feat)

        # 均值和标准差分别比较
        loss += nn.MSELoss()(gen_mean, style_mean)
        loss += nn.MSELoss()(gen_std,  style_std)
    return loss


def tv_loss(img):
    """
    全变分损失：惩罚相邻像素差异，抑制棋盘格等高频噪声
    img: (B, C, H, W)
    """
    h_diff = img[:, :, 1:, :] - img[:, :, :-1, :]
    w_diff = img[:, :, :, 1:] - img[:, :, :, :-1]
    return torch.mean(torch.abs(h_diff)) + torch.mean(torch.abs(w_diff))
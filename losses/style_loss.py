# -*- coding: utf-8 -*-
import torch

def gram_matrix(feature_map):
    B, C, H, W = feature_map.shape
    M = H * W
    F = feature_map.view(B, C, M)
    G = torch.bmm(F, F.transpose(1, 2))
    return G   # 除以C*M，和官方一致

def style_loss(generated_features, style_features):
    loss = 0
    w = 1.0 / len(generated_features)

    for gen_f, style_f in zip(generated_features, style_features):
        G = gram_matrix(gen_f)
        A = gram_matrix(style_f)
        loss += w * torch.nn.functional.mse_loss(G, A, reduction='mean')

    return loss
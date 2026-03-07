# losses/content_loss.py
import torch

def content_loss(generated_feature, content_feature):
    """
    对应论文公式：
    L_content = 1/2 * sum((F - P)^2)
    F: 生成图conv4_2特征图
    P: 内容图conv4_2特征图
    """
    return 0.5 * torch.mean((generated_feature - content_feature) ** 2)
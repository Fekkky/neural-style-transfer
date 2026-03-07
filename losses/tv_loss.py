import torch

def total_variation_loss(img):
    """
    TV Loss：让相邻像素之间的差异尽量小，生成图更平滑
    水平方向差异 + 垂直方向差异
    """
    # 水平方向：相邻列之间的差
    diff_h = img[:, :, :, 1:] - img[:, :, :, :-1]
    # 垂直方向：相邻行之间的差
    diff_v = img[:, :, 1:, :] - img[:, :, :-1, :]

    return torch.mean(torch.abs(diff_h)) + torch.mean(torch.abs(diff_v))
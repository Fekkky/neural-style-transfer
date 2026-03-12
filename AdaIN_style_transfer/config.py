# -*- coding: utf-8 -*-
class Config:
    # 数据路径
    content_dir   = "../data/content/Adain_content/train2017"
    style_dir     = "../data/style/"
    output_dir    = "../data/output/AdaIN_improved/"

    # 训练参数
    image_size    = 256
    batch_size    = 8
    num_epochs    = 5
    lr            = 1e-4
    

    # 损失权重
    content_weight = 1.0
    style_weight   = 10.0
    tv_weight      = 1e-4

    # 推理参数
    alpha         = 1.0        # 风格强度 0~1
    content_img   = "../data/content/Gatys_content/content2.jpg"
    style_img     = "../data/style/WaterLilies.jpg"

    # 模型保存
    save_path     = "checkpoints/adain.pth"
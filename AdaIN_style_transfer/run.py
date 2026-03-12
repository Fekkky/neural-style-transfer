# -*- coding: utf-8 -*-
import torch
import os
from config import Config
from models.encoder import VGGEncoder
from models.decoder_improved import Decoder
from models.adain import AdaIN
from utils.image_utils import load_image, save_image, show_images

def run():
    cfg    = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    encoder = VGGEncoder().to(device)
    decoder = Decoder().to(device)
    adain   = AdaIN().to(device)
    decoder.load_state_dict(torch.load(cfg.save_path, map_location=device))
    decoder.eval()

    # 加载图片
    content = load_image(cfg.content_img, cfg.image_size).to(device)
    style   = load_image(cfg.style_img,   cfg.image_size).to(device)

    with torch.no_grad():
        # 提取特征
        content_feats = encoder(content)
        style_feats   = encoder(style)

        # AdaIN风格迁移
        t = adain(content_feats[-1], style_feats[-1])

        # alpha控制风格强度
        t = cfg.alpha * t + (1 - cfg.alpha) * content_feats[-1]

        # 生成图像
        generated = decoder(t, content_feats)

    # 保存结果
    content_name = os.path.splitext(os.path.basename(cfg.content_img))[0]
    style_name   = os.path.splitext(os.path.basename(cfg.style_img))[0]
    output_dir   = os.path.join(cfg.output_dir, content_name)
    os.makedirs(output_dir, exist_ok=True)
    save_image(generated, os.path.join(output_dir, f"{style_name}.jpg"))

    show_images(content, style, generated)

if __name__ == '__main__':
    run()

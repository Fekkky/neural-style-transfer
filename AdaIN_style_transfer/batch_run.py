# -*- coding: utf-8 -*-
"""
批量风格迁移脚本
遍历 Gatys_content 所有内容图 × style 所有风格图，逐一生成
输出结构：
    ../data/output/AdaIN/
        <内容图名>/
            <风格图名>.jpg
"""
import os
import torch
from itertools import product

from config import Config
from models.encoder import VGGEncoder
from models.decoder_improved import Decoder
from models.adain import AdaIN
from utils.image_utils import load_image, save_image


def batch_run():
    cfg    = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Batch] 使用设备：{device}")

    # ── 加载模型 ─────────────────────────────────────────────
    encoder = VGGEncoder().to(device)
    decoder = Decoder().to(device)
    adain   = AdaIN().to(device)

    assert os.path.exists(cfg.save_path), \
        f"模型文件不存在：{cfg.save_path}，请先运行 train.py"

    decoder.load_state_dict(torch.load(cfg.save_path, map_location=device))
    encoder.eval()
    decoder.eval()

    # ── 路径配置 ─────────────────────────────────────────────
    content_dir = "../data/content/Gatys_content"
    style_dir   = "../data/style"
    output_root = "../data/output/AdaIN_improved"

    exts = ('.jpg', '.jpeg', '.png')

    content_files = sorted([
        os.path.join(content_dir, f)
        for f in os.listdir(content_dir)
        if f.lower().endswith(exts)
    ])
    style_files = sorted([
        os.path.join(style_dir, f)
        for f in os.listdir(style_dir)
        if f.lower().endswith(exts)
    ])

    total = len(content_files) * len(style_files)
    print(f"[Batch] 内容图 {len(content_files)} 张 × 风格图 {len(style_files)} 张 = {total} 组合")
    print(f"[Batch] 开始生成...\n")

    # ── 批量推理 ─────────────────────────────────────────────
    count = 0
    for content_path in content_files:
        content_name = os.path.splitext(os.path.basename(content_path))[0]

        # 以内容图名为父目录
        save_dir = os.path.join(output_root, content_name)
        os.makedirs(save_dir, exist_ok=True)

        # 加载内容图（每张内容图只加载一次，所有风格图共用）
        content = load_image(content_path, cfg.image_size).to(device)

        with torch.no_grad():
            content_feats = encoder(content)

        for style_path in style_files:
            style_name  = os.path.splitext(os.path.basename(style_path))[0]
            output_path = os.path.join(save_dir, f"{style_name}.jpg")

            # 已存在则跳过，方便断点续跑
            if os.path.exists(output_path):
                print(f"  [跳过] {content_name} × {style_name} 已存在")
                count += 1
                continue

            style = load_image(style_path, cfg.image_size).to(device)

            with torch.no_grad():
                style_feats = encoder(style)
                t           = adain(content_feats[-1], style_feats[-1])
                t           = cfg.alpha * t + (1 - cfg.alpha) * content_feats[-1]
                generated   = decoder(t, content_feats)

            save_image(generated, output_path)
            count += 1
            print(f"  [{count:3d}/{total}] {content_name} × {style_name} → {output_path}")

    print(f"\n[Batch] 全部完成，共生成 {count} 张图像")


if __name__ == '__main__':
    batch_run()
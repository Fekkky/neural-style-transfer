# -*- coding: utf-8 -*-
import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from config import Config
from models.encoder import VGGEncoder
from models.decoder_improved import Decoder
from models.adain import AdaIN
from losses.loss import content_loss, style_loss, tv_loss
from utils.image_utils import ContentDataset, load_image


def train():
    cfg    = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] 使用设备：{device}")

    encoder = VGGEncoder().to(device)
    decoder = Decoder().to(device)
    adain   = AdaIN().to(device)
    encoder.eval()

    optimizer = optim.Adam(decoder.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True
    )

    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    content_dataset = ContentDataset(cfg.content_dir, transform=transform)
    content_loader  = DataLoader(
        content_dataset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )

    # 预加载风格图到内存
    style_files = [
        os.path.join(cfg.style_dir, f)
        for f in os.listdir(cfg.style_dir)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]
    style_tensors = [load_image(p, cfg.image_size).to(device) for p in style_files]
    print(f"[Train] 共加载 {len(style_tensors)} 张风格图")



    for epoch in range(cfg.num_epochs):
        decoder.train()
        epoch_loss = 0.0

        for i, content in enumerate(content_loader):
            content = content.to(device)
            
            # 每步随机取一张风格图，临时算特征，用完自动释放
            style = random.choice(style_tensors)
            style_expand = style.expand(content.size(0), -1, -1, -1)
            
            with torch.no_grad():
                content_feats = encoder(content)
                style_feats   = encoder(style_expand)

            t = adain(content_feats[-1], style_feats[-1])
            generated = decoder(t, content_feats)
            generated_feats = encoder(generated)  # no_grad 外，梯度能回传

            c_loss = content_loss(generated_feats[-1], t.detach())
            s_loss = style_loss(generated_feats, style_feats)
            t_loss  = tv_loss(generated)
            loss    = cfg.content_weight * c_loss + cfg.style_weight * s_loss + cfg.tv_weight * t_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch} | Step {i:4d} | "
                      f"Loss={loss.item():.4f} | "
                      f"Content={c_loss.item():.4f} | "
                      f"Style={s_loss.item():.4f}")

        avg_loss = epoch_loss / len(content_loader)
        scheduler.step(avg_loss)
        print(f"[Epoch {epoch}] avg_loss={avg_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")

    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    torch.save(decoder.state_dict(), cfg.save_path)
    print(f"[Train] 模型已保存到 {cfg.save_path}")


if __name__ == '__main__':
    train()
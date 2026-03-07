import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os


MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def load_image(path, size=512):
    """读取图片并预处理成tensor"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    img = Image.open(path).convert("RGB")
    # unsqueeze(0)加上batch维度 → [1, 3, H, W]
    return transform(img).unsqueeze(0)

def save_image(tensor, path):
    """把tensor反归一化后保存为图片"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 反归一化
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)

    img = tensor.clone().squeeze(0).cpu()  # 去掉batch维度
    img = img * std + mean                 # 反归一化
    img = img.clamp(0, 1)                  # 截断到[0,1]

    # 转成PIL图片保存
    img = transforms.ToPILImage()(img)
    img.save(path)
    print(f"图片已保存到 {path}")

def show_images(content, style, generated):
    """并排展示三张图片"""
    def to_pil(tensor):
        mean = torch.tensor(MEAN).view(3, 1, 1)
        std  = torch.tensor(STD).view(3, 1, 1)
        img  = tensor.clone().squeeze(0).cpu()
        img  = img * std + mean
        img  = img.clamp(0, 1)
        return transforms.ToPILImage()(img)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(to_pil(content));   axes[0].set_title("内容图"); axes[0].axis("off")
    axes[1].imshow(to_pil(style));     axes[1].set_title("风格图"); axes[1].axis("off")
    axes[2].imshow(to_pil(generated)); axes[2].set_title("生成图"); axes[2].axis("off")
    plt.tight_layout()
    plt.show(),
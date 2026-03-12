# -*- coding: utf-8 -*-
import torch
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class ContentDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform   = transform
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img
    

class StyleDataset(Dataset):
    """风格图像数据集，用于多风格训练"""
    def __init__(self, root_dir, image_size=256):
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')
        ]
        if not self.image_paths:
            raise ValueError(f"风格目录 {root_dir} 为空或没有有效图像文件")
        
        print(f"已加载 {len(self.image_paths)} 张风格图像 from {root_dir}")
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img 
    
# ImageNet均值和标准差
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def load_image(path, size=256):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)  # 加batch维度

def save_image(tensor, path):
    # 反归一化
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    img  = tensor.squeeze(0).cpu() * std + mean
    img  = img.clamp(0, 1)
    transforms.ToPILImage()(img).save(path)

def show_images(content, style, generated):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['内容图', '风格图', '生成图']
    for ax, img, title in zip(axes, [content, style, generated], titles):
        mean = torch.tensor(MEAN).view(3, 1, 1)
        std  = torch.tensor(STD).view(3, 1, 1)
        img  = img.squeeze(0).cpu() * std + mean
        img  = img.clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.show()
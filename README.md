基于神经风格迁移的图像艺术化处理系统
基于 Gatys et al. 2015 论文《A Neural Algorithm of Artistic Style》的 PyTorch 实现，通过优化生成图像的像素值，使其同时保留内容图的语义结构和风格图的纹理特征。

效果展示
生成效果案例1
![alt text](image-1.png)
![alt text](image-3.png)
![alt text](image-2.png)
生成效果案例2
![alt text](image-5.png)
![alt text](image-6.png)
![alt text](image-4.png)
生成效果案例3
![alt text](image-7.png)
![alt text](image-8.png)
![alt text](image-9.png)
算法原理
核心思想
本项目使用预训练的 VGG-19 作为特征提取器，通过梯度下降优化生成图像的像素值，使其同时满足两个约束：

内容约束：生成图在深层特征上与内容图相似
风格约束：生成图在多层 Gram 矩阵上与风格图相似


优化过程
与普通神经网络训练不同，本方法固定 VGG-19 权重，优化生成图的像素值本身：
内容图 → VGG-19 → conv4_2特征图（固定）
风格图 → VGG-19 → 五层Gram矩阵（固定）
生成图（初始=内容图副本）
    ↓ 每轮迭代
    ├─ 提取特征 → 计算三个损失
    ├─ 反向传播 → 对像素求梯度
    └─ L-BFGS  → 更新像素值

项目结构
Artistic Style Transfer/
├── config.py                # 超参数配置
├── run.py                   # 主程序入口
├── requirements.txt         # 依赖库
├── models/
│   ├── __init__.py
│   └── vgg_extractor.py     # VGG-19特征提取器
├── losses/
│   ├── __init__.py
│   ├── content_loss.py      # 内容损失
│   ├── style_loss.py        # 风格损失（含Gram矩阵）
│   └── tv_loss.py           # 总变差损失
├── utils/
│   ├── __init__.py
│   └── image_utils.py       # 图片读取、归一化、保存
└── data/
    ├── content/             # 内容图片
    ├── style/               # 风格图片
    └── output/              # 生成结果（按内容图分目录）

环境安装
1. 创建虚拟环境
bashconda create -n style_transfer python=3.10
conda activate style_transfer
2. 安装 PyTorch（CUDA 11.8）
bashpip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

其他 CUDA 版本请前往 pytorch.org 获取对应安装命令

3. 安装其余依赖
bashpip install -r requirements.txt
4. 验证安装
pythonimport torch
print(torch.cuda.is_available())    # True
print(torch.cuda.get_device_name(0))

使用方法
1. 准备图片
将内容图放入 data/content/，风格图放入 data/style/，支持 .jpg、.png 等常见格式。
2. 修改配置
编辑 config.py：
pythoncontent_img = "data/content/你的内容图.jpg"
style_img   = "data/style/你的风格图.jpg"
3. 运行
bashconda activate style_transfer
python run.py
生成结果自动保存到 data/output/内容图名称/风格图名称.png。

参数说明
参数默认值说明image_size400图片缩放尺寸，越大效果越好但越慢num_steps1000优化迭代次数，L-BFGS 下 1000 步通常足够optimizerlbfgs优化器，推荐 lbfgs，也支持 adamcontent_weight1e5内容损失权重 αstyle_weight3e4风格损失权重 βtv_weight1e0TV 损失权重 γ，控制平滑程度lr1.0学习率（仅 Adam 有效）
调参建议
风格太弱，内容太明显  →  降低 content_weight 或提高 style_weight
风格太强，内容模糊    →  提高 content_weight 或降低 style_weight
生成图噪点多          →  提高 tv_weight
生成图过于平滑        →  降低 tv_weight

实验观察
风格图与内容图语义相关性的影响
实验发现，当内容图与风格图语义相似时（如同为花卉题材），风格迁移效果显著优于语义差异较大的组合。
原因分析： Gram 矩阵只捕捉特征的共现关系，不保留空间位置信息。当内容图与风格图语义相近时，VGG 提取的特征在空间分布上接近，内容损失与风格损失的优化方向基本一致，迁移效果自然。反之，两个损失优化方向冲突，生成图出现杂乱现象。
各超参数对结果的影响
变量观察结果content_weight ↑内容结构更清晰，风格减弱style_weight ↑风格纹理更明显，内容趋于模糊tv_weight ↑图像更平滑，噪点减少num_steps ↑损失持续下降，细节更精细

参考文献
Gatys, L. A., Ecker, A. S., & Bethge, M. (2015).
A Neural Algorithm of Artistic Style.
arXiv:1508.06576

开发环境

Python 3.10
PyTorch 2.1.1 + CUDA 11.8
NVIDIA GeForce RTX 3060 Laptop GPU
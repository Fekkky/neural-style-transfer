# 神经风格迁移

基于 PyTorch 实现的三种神经风格迁移方法：**Gatys 优化方法**、**AdaIN 实时任意风格迁移**，以及本项目的核心改进——**引入跳跃连接的 AdaIN**，从根本上解决了棋盘格伪影问题。



---

## 效果展示

| 内容图 | 风格图 | 输出结果（AdaIN + 跳跃连接）|
|--------|--------|--------------------------|
| content2.jpg | 星夜 | *(在此插入效果图)* |
| mountain.jpg | 吻 | *(在此插入效果图)* |

---

## 方法介绍

### 1. Gatys 优化方法
Gatys 等人（2015）提出的原始神经风格迁移方法。通过 L-BFGS 对合成图像进行迭代优化，最小化内容损失与基于 Gram Matrix 的风格损失的加权组合。

- 生成质量最高
- 每张图约 3 分钟（300 次迭代，RTX 3060）
- 每次只能迁移一种风格

### 2. AdaIN 实时任意风格迁移
Huang & Belongie（2017）提出的前馈网络方法。将内容特征的逐通道均值和方差对齐至风格特征的统计量，AdaIN 层本身无任何可训练参数。

- 单张 256×256 图像推理约 20ms
- 无需重新训练即可支持任意风格
- 使用 MS-COCO train2017 + 23 张艺术风格图训练

### 3. AdaIN + 跳跃连接（本项目改进）
原始 AdaIN 解码器需将 16×16 的特征图上采样 16 倍，在高频纹理风格（如星夜、吻）上会产生明显的**棋盘格伪影**。改进型解码器将编码器浅层特征直接注入解码器的对应分辨率层，从根本上弥补大倍率上采样导致的空间细节损失。

```
编码器：relu1_1 (H×W,   64ch)  ──────────────────────────→ concat
        relu2_1 (H/2,  128ch)  ──────────────→ concat
        relu3_1 (H/4,  256ch)  ──────→ concat
        relu4_1 (H/8,  512ch)  → AdaIN → ↑2× → Fusion3 → ↑2× → Fusion2 → ↑2× → Fusion1 → ↑2× → RGB
```

其他改进：
- `ReflectionPad2d` 替代零填充 → 彻底消除边缘色彩扩散伪影
- 全变分损失（TV Loss，λ=1e-4）→ 生成图像更平滑
- 训练时从 23 张风格图中随机采样 → 提升模型泛化能力

---

## 项目结构

```
neural-style-transfer/
├── Gatys/
│   └── gatys.py                   # 优化方法实现
│
└── AdaIN_style_transfer/
    ├── config.py                  # 超参数集中配置
    ├── train.py                   # 训练脚本
    ├── run.py                     # 单张图像推理
    ├── batch_run.py               # 批量推理
    ├── models/
    │   ├── encoder.py             # VGG-19 截至 relu4_1（权重固定）
    │   ├── decoder.py             # 带跳跃连接的解码器
    │   └── adain.py               # AdaIN 层（无可训练参数）
    ├── losses/
    │   └── loss.py                # 内容损失 + 风格损失 + TV 损失
    └── utils/
        └── image_utils.py
```

---

## 快速开始

### 环境配置

```bash
conda create -n nst python=3.10
conda activate nst
pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install pillow tqdm
```

### Gatys 方法（无需训练）

```bash
cd Gatys
python gatys.py \
  --content ../data/content/content2.jpg \
  --style   ../data/style/theStarNight.jpg \
  --output  output.jpg \
  --iters   300
```

### AdaIN 推理（使用预训练权重）

```bash
cd AdaIN_style_transfer
python run.py \
  --content ../data/content/content2.jpg \
  --style   ../data/style/theStarNight.jpg \
  --output  ../data/output/AdaIN/result.jpg
```

### AdaIN 从头训练

1. 下载 [MS-COCO train2017](https://cocodataset.org/#download)，放至 `data/content/Adain_content/train2017/`
2. 将风格图放至 `data/style/`
3. 执行训练：

```bash
cd AdaIN_style_transfer
python train.py
```

`config.py` 中的主要参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `image_size` | 256 | 训练分辨率 |
| `batch_size` | 8 | 针对 6GB 显存调整 |
| `num_epochs` | 2 | train2017 约 29,500 步/epoch |
| `style_weight` | 10.0 | 风格损失权重（内容权重为 1.0）|
| `tv_weight` | 1e-4 | 全变分正则化强度 |

---

## 关键设计决策

**为什么需要跳跃连接？**
AdaIN 在 `relu4_1`（分辨率 16×16）处完成风格变换，解码器需将其上采样 16 倍还原至原始分辨率。在此过程中大量空间细节丢失，在高频纹理风格图上产生周期性棋盘格伪影。跳跃连接将编码器浅层特征直接传入解码器对应层，从根本上解决了这一问题。

**为什么使用反射填充？**
零填充在图像边缘引入人工边界，卷积响应在边缘产生系统性偏差，导致色彩向外扩散。反射填充以边缘像素为轴进行镜像延伸，保持特征连续性，彻底消除边缘伪影。

**为什么不预计算风格特征？**
23 张风格图在 `relu4_1` 产生的特征约 37MB，加上训练中间张量，6GB 显存很快耗尽（OOM）。最终方案是将风格图以 CPU 张量形式存储，每步随机取一张搬至 GPU，用完即释放。


---

## 实验环境

| 组件 | 版本 |
|------|------|
| GPU | NVIDIA RTX 3060 Laptop（6GB）|
| CUDA | 11.8 |
| Python | 3.10 |
| PyTorch | 2.1.1+cu118 |

---

## 参考文献

- Gatys et al. — [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)（2015）
- Johnson et al. — [Perceptual Losses for Real-Time Style Transfer](https://arxiv.org/abs/1603.08155)（2016）
- Huang & Belongie — [Arbitrary Style Transfer in Real-time with AdaIN](https://arxiv.org/abs/1703.06868)（2017）
- Odena et al. — [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/)（2016）

---

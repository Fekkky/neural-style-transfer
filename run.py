import torch
import os
from torch.optim import Adam, LBFGS
from tqdm import tqdm

from config import Config
from models.vgg_extractor import VGGFeatureExtractor
from losses.content_loss import content_loss
from losses.style_loss import style_loss
from losses.tv_loss import total_variation_loss
from utils.image_utils import load_image, save_image, show_images

def main():
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    content_img = load_image(cfg.content_img, cfg.image_size).to(device)
    style_img   = load_image(cfg.style_img,   cfg.image_size).to(device)
    print("图片加载完成")

    vgg = VGGFeatureExtractor().to(device)
    print("VGG加载完成")

    with torch.no_grad():
        content_features = vgg(content_img)
        style_features   = vgg(style_img)

    # index 4 = conv4_2 内容层
    # index 0,1,2,3,5 = 风格层
    content_idx = 4
    style_idx   = [0, 1, 2, 3, 5]

    generated = content_img.clone().requires_grad_(True)

    if cfg.optimizer == "lbfgs":
        optimizer = LBFGS([generated],
                        max_iter=cfg.num_steps,
                        line_search_fn='strong_wolfe')
        cnt = [0]

        def closure():
            optimizer.zero_grad()
            gen_features = vgg(generated)

            loss_c  = content_loss(gen_features[content_idx], content_features[content_idx])
            loss_s  = style_loss([gen_features[i] for i in style_idx], [style_features[i] for i in style_idx])
            loss_tv = total_variation_loss(generated)

            loss = (cfg.content_weight * loss_c +
                    cfg.style_weight   * loss_s +
                    cfg.tv_weight      * loss_tv)

            loss.backward()

            if cnt[0] % 50 == 0:
                print(f"step {cnt[0]:4d} | "
                    f"total={loss.item():.2f} | "
                    f"content={cfg.content_weight * loss_c.item():.2f} | "
                    f"style={cfg.style_weight * loss_s.item():.2f} | "
                    f"tv={cfg.tv_weight * loss_tv.item():.4f}")
            cnt[0] += 1
            return loss

        optimizer.step(closure)   # 只调用一次，内部自动跑num_steps次

    else:  # adam
        optimizer = Adam([generated], lr=cfg.lr)
        for step in tqdm(range(cfg.num_steps)):
            optimizer.zero_grad()
            gen_features = vgg(generated)

            loss_c  = content_loss(gen_features[content_idx], content_features[content_idx])
            loss_s  = style_loss([gen_features[i] for i in style_idx], [style_features[i] for i in style_idx])
            loss_tv = total_variation_loss(generated)

            loss = (cfg.content_weight * loss_c +
                    cfg.style_weight   * loss_s +
                    cfg.tv_weight      * loss_tv)

            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                tqdm.write(f"step {step:4d} | total={loss.item():.2f} | "
                           f"content={cfg.content_weight*loss_c.item():.2f} | "
                           f"style={cfg.style_weight*loss_s.item():.2f}")

    # 根据内容图文件名生成目录
    content_name = os.path.splitext(os.path.basename(cfg.content_img))[0]
    style_name   = os.path.splitext(os.path.basename(cfg.style_img))[0]
    output_dir   = os.path.join(cfg.output_dir, content_name)
    os.makedirs(output_dir, exist_ok=True)

    # 以风格图命名生成结果，避免覆盖
    save_image(generated, os.path.join(output_dir, f"{style_name}.jpg"))
    show_images(content_img, style_img, generated)  
if __name__ == "__main__":
    main()
class Config:
    content_img    = "data/content/content1.jpg"
    style_img      = "data/style/WaterLilies.jpg"
    output_dir     = "data/output/"

    image_size     = 400
    num_steps      = 2000     # lbfgs 1000步够用
    optimizer      = "lbfgs"    # 切回lbfgs

    content_weight = 1.6
    style_weight   = 1e-4
    tv_weight      = 1e-3
    lr             = 1.0        # lbfgs不用lr，这个值没影响
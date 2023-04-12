import imageio
from imgaug import augmenters as iaa
import numpy as np
import os
import pandas as pd
import glob

path="/mnt/llz/dataset/Project_npc/work_dir/t2/train/npc/"
src_paths=glob.glob(os.path.join(path,"*"))
for i,src_path in enumerate(src_paths):
    image = imageio.imread(src_path)
    src_img_name = src_path.split('/')[-1].split('.')[0]

    images = [image, image, image, image, image, image]
    # images = [image, image, image]

    seq = iaa.SomeOf(1,[
        # iaa.Affine(rotate=(-25, 25)),
        iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # 缩放
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # 平移
        rotate=(-20, 60),  # 旋转
        shear=(-8, 8)),
        iaa.Fliplr(1.0),
        iaa.Flipud(0.5),
        iaa.CropAndPad(percent=(-0.25, 0.25)),
        iaa.ContrastNormalization((0.75, 1.5)),
        iaa.GaussianBlur(sigma=0.2),
        iaa.AverageBlur(k=2),
        iaa.MedianBlur(k=3),
        iaa.MotionBlur(),
        iaa.GammaContrast(gamma=1.5),
        # iaa.SigmoidContrast(gain=3),
        # iaa.LogContrast(gain=5),
        # iaa.LinearContrast(alpha=10),
        # iaa.Crop(percent=(0, 0.2), keep_size=True)
    ])

    images_aug = seq(images=images)

    for index, image in enumerate(images_aug):
        aug_path_name = os.path.join(path, "{}_aug_{}.png".format(src_img_name, index))
        print("{} / {} size is{}".format(i,len(src_paths),aug_path_name, image.shape))
        imageio.imwrite(aug_path_name, image)

    # print(ori_path_name)

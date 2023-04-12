import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Compose,Resize,ToTensor
from einops import rearrange,reduce,repeat
import os


def crop_patch(image_path,save_path,patch_size=16,resize_224x224=False):
    """

    :param image_path:输入裁剪图片的路径，例如 '/mnt/llz/media/images/test.jpg'
    :param save_path: 输入裁剪后图片的路径，例如：'/mnt/llz/media/images/patches/'
    :param patch_size: 裁剪图片的大小
    :param resize_224x224: 将输入图片的大小改变，要注意图片大小必须被patch_size整除
    :return:
    """
    img=Image.open(image_path)
    if resize_224x224:
        transform=Compose([Resize((224,224)),ToTensor()])
    else:
        transform = Compose([ToTensor()])
    x=transform(img)
    patches=rearrange(x,' c (h s1) (w s2) -> c s1 s2 (h w)',s1=patch_size,s2=patch_size)
    patches=patches.numpy().transpose(3,1,2,0)
    print('分为',patches.shape[0],'张图片')
    for i in range(patches.shape[0]):
        #在此设置crop后图片的名称
        plt.imsave(os.path.join(save_path,i+'.jpg'),patches[i])


if __name__ == '__main__':
    save_path='/mnt/llz/media/images/patches/'
    image_path='/mnt/llz/media/images/test.jpg'
    crop_patch(image_path,save_path)



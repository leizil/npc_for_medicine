import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

#读取nii文件
def read_niifile(niifile):  # 读取niifile文件
    img = nib.load(niifile)  # 下载niifile文件（其实是提取文件）
    img_fdata = img.get_fdata()  # 获取niifile数据
    img90 = np.rot90(img_fdata) #旋转90度
    #return img_fdata
    return img90

# output='.'
#保存jpg文件并输出
def save_fig(file,png_path):  # 保存为图片
    name=file.split('\\')[-1].split('.')[0]
    print(name)
    file_path=os.path.join(png_path,name)
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    fdata = read_niifile(file)  # 调用上面的函数，获得数据
    (y, x, z) = fdata.shape  # 获得数据shape信息：（长，宽，维度-即切片数量）
    for k in range(z):
        silce = fdata[:, :, k]
        #silce = fdata[k, :, :]  # 三个位置表示三个不同角度的切片
        plt.imsave(os.path.join(file_path, '{}.png'.format(str(k))), silce,cmap='gray')
        # 将切片信息保存为png格式
        #str(k)代表每层切片单独命名，避免重名，以_0,_1,...的形式命名


def test(nii_path,png_path):
    # png_path=r'D:\code\算法\lab_server_code\data\png'
    # nii_path=r'D:\code\算法\lab_server_code\data\nii\t1_se_1.nii.gz'
    save_fig(nii_path,png_path)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nii_path', default=r'D:\code\算法\lab_server_code\data\nii\t1_se_1.nii.gz')
    parser.add_argument('--png_path', default=r'D:\code\算法\lab_server_code\data\png')
    return parser


if __name__ == '__main__':
    parser=get_parser()
    args=parser.parse_args()
    nii_path=args.nii_path
    png_path=args.png_path
    test(nii_path,png_path)
    print(nii_path)
    print(png_path)


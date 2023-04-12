import pandas as pd
import torch
import nibabel as nib
import numpy as np
from  PIL import Image
import sys
sys.path.append('/mnt/llz/code/cls/mymodel/utiles')
import image_csv
from torchvision.transforms import CenterCrop
import time
import random
import tqdm


import os
import shutil


def crop_img(df,is_train=True,base_crop_img='/mnt/llz/media/myNpcDiagnoseProjectDataset/cropData'):
    path_lists=df['path'].tolist()
    if is_train:
        crop_path=os.path.join(base_crop_img,'train')
        name='trainInfo.csv'
    else:
        crop_path=os.path.join(base_crop_img,'test')
        name='testInfo.csv'
    if not os.path.exists(crop_path):
        os.mkdir(crop_path)
        for img_path in path_lists:
            img = Image.open(img_path)
            img_name = img_path.split('/')[-1]
            crop_img = CenterCrop((200, 200))
            img = crop_img(img)
            img = img.resize((224, 224))
            crop_img_path = os.path.join(crop_path, img_name)
            img.convert("L")
            img.save(crop_img_path, format='PNG')
            s1 = df['path'] == img_path
            idx = 0
            for b in s1.values:
                if b == True:
                    break
                idx += 1
            print(idx, '---------->', crop_img_path)
            df['crop_path']=df['crop_path'].astype('string')
            # print(df['crop_path'].dtype)
            df.at[idx, 'crop_path'] = crop_img_path
            # print(idx,'---------->',crop_img_path)
        df.to_csv('/mnt/llz/media/myNpcDiagnoseProjectDataset/'+name,index=None)
    else:
        print(base_crop_img,'exists! ')


    print('crop_path fixed')

def load_img(path,df):
    img=Image.open(path)
    img=img.convert('RGB')
    img = img.resize((224, 224))
    data = np.array(img, dtype=np.float32)
    if len(data.shape)==2:
        data0=np.expand_dims(data,axis=2)
        data1=data0
        data2=data0
        data=np.concatenate((data0,data1,data2),axis=2)
    data = np.transpose(data, (2, 0, 1))
    if data.shape !=(3,224,224):
        print(path)
    return data




class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None,med=False):
        self.df = df
        self.label = label
        # print(df.columns)
        self.paths = df['path'].tolist()
        self.clses = df['cls_no'].tolist()
        self.transforms = transforms
        self.med=med
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path = self.paths[index]
        if self.med :
            data = load_niigz(path, self.df)
        else:
            data=load_img(path,self.df)
        if self.transforms:
            # data = self.transforms(image=niigz)
            niigz = data['image']


        label=torch.tensor(self.clses[index])
        return torch.tensor(data),label






def myDataSet(df):
    mydataset=BuildDataset(df)
    return mydataset

def test():
    df=image_csv.read_my_dir()
    print('df:',df.shape)
    mydataset=myDataSet(df)
    n=0
    for _,y in mydataset:
        n+=1
        if n==1:
            print(_.shape)
    print("共有 ",n," 个数据")

if __name__ == '__main__':
    test()



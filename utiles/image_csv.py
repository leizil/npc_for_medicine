import os
import pandas as pd
import numpy as np
import nibabel as nib
import torch
from PIL import Image
import torch.nn as nn
import torch.functional as F
from sklearn.model_selection import StratifiedKFold
import json

import sys
sys.path.append('/mnt/llz/code/cls/mymodel/utiles')
import get_json




def read_img_data(img_dir,is_nii=False):
    if is_nii:
        data = nib.load(img_dir)
        img = data.get_fdata()
        return img
    else:
        data=Image.open(img_dir)
        data=np.array(data,dtype=np.float32)

        if len(data.shape) == 2:
            data=np.expand_dims(data,axis=2)
            # print(data.shape)
        return data

def get_df():
    imgs_path=[]
    #图片类名的编号
    cls_no = []
    #一类图片的名称
    cls_name=[]
    df = pd.DataFrame()
    df['cls_no'] = cls_no
    df['cls_name']=cls_name
    df['crop_path']='null'
    return df
df=get_df()

def read_my_dir():
    datadict=get_json.open_json()
    n_splits=datadict['n_splits']
    data_dir = datadict['data_dir']
    csv_dir = datadict['csv_dir']
    test_dir=datadict['test_path']
    save_csv_dir=os.path.join(csv_dir, 'trainInfo.csv')
    save_test_dir=os.path.join(csv_dir,'testInfo.csv')

    if os.path.exists(save_csv_dir) and os.path.exists(save_test_dir):
        print(save_csv_dir,' exists  and df returns !')
        print(save_test_dir,' test exists  and df returns !')
        return pd.read_csv(save_csv_dir),pd.read_csv(save_test_dir)


    classes_name = os.listdir(data_dir)
    classes_name_test = os.listdir(test_dir)
    print('find {} classes'.format(len(classes_name)))
    df = get_df()
    df_test=get_df()
    for i in range(len(classes_name)):
        clsDir = os.path.join(data_dir, classes_name[i])
        # 每一类图片 名称class_name
        class_name = os.listdir(clsDir)
        imgs_path = []
        # 图片类名的编号
        cls_no = []
        # 一类图片的名称
        cls_name = []

        print('get{}  {} \t'.format(i, clsDir),end='\n')
        for img_name in class_name:
            # 将图片类名添加到列表
            cls_name.append(classes_name[i])
            # 将图片类编号添加到列表
            cls_no.append(i)
            # allList.append(clslist)
            # 将图片路径添加到列表

            img_path = os.path.join(clsDir, img_name)
            imgs_path.append(img_path)
            # img = read_img_data(img_path)
            # shape.append(img.shape)
            # channel.append(img.shape[2])
            # width.append(img.shape[0])
            # height.append(img.shape[1])
        new_df = pd.DataFrame(imgs_path, columns=['path'])
        new_df['cls_no'] = cls_no
        new_df['cls_name'] = cls_name
        # new_df['shape'] = shape
        # new_df['width'] = width
        # new_df['height'] = height
        # new_df['channels'] = channel
        df = df.append(new_df, ignore_index=True)
    for i in range(len(classes_name_test)):
        clsDir = os.path.join(test_dir, classes_name_test[i])
        # print('test: clsDir: ',clsDir)
        # print('test: test_dir: ', test_dir)
        # print('test: classes_name_test: ', classes_name_test[i])
        # 每一类图片 名称class_name
        class_name = os.listdir(clsDir)
        # print('test: class_name: ', class_name)
        imgs_path = []
        # 图片类名的编号
        cls_no = []
        # 一类图片的名称
        cls_name = []

        print('get{}  {} \t'.format(i, clsDir),end='\n')
        for img_name in class_name:
            # 将图片类名添加到列表
            cls_name.append(classes_name[i])
            # 将图片类编号添加到列表
            cls_no.append(i)
            # allList.append(clslist)
            # 将图片路径添加到列表

            img_path = os.path.join(clsDir, img_name)
            print(img_path)
            imgs_path.append(img_path)
        new_df = pd.DataFrame(imgs_path, columns=['path'])
        new_df['cls_no'] = cls_no
        new_df['cls_name'] = cls_name

        df_test = df_test.append(new_df, ignore_index=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df['path'], df['cls_no'])):
        df.loc[val_idx, 'fold'] = fold
        # display(df.groupby(['fold', 'cls'])['name'].count())
    print('done!  csv file is in ',save_csv_dir)
    print('done!  test csv file is in ', save_test_dir)
    df.to_csv(save_csv_dir, index=None)
    df_test.to_csv(save_test_dir,index=None)
    return df,df_test


if __name__ == '__main__':
    read_my_dir()


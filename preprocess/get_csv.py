import os
import pandas as pd
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.functional as F
from sklearn.model_selection import StratifiedKFold
import json

import sys
sys.path.append('/mnt/llz/code/cls/mymodel/utiles')
import get_json


def read_my_dir():
    datadict=get_json.open_json()
    n_splits=datadict['n_splits']
    data_dir = datadict['data_dir']
    csv_dir = datadict['csv_dir']

    save_csv_dir=os.path.join(csv_dir, 'dataInfo.csv')

    if os.path.exists(save_csv_dir):
        print(save_csv_dir,' exists  and df returns !')
        return pd.read_csv(save_csv_dir)

    classes = os.listdir(data_dir)
    if len(classes) == 2:
        print('find 2 classes', classes[0], ' ', classes[1])
        cls1Dir = os.path.join(data_dir, classes[0])
        cls2Dir = os.path.join(data_dir, classes[1])
        cls1lists = os.listdir(cls1Dir)
        cls2lists = os.listdir(cls2Dir)
        cls1listsDir = []
        cls1No = []
        cls2No = []
        allList = []
        size = []
        cls_no = []
        n_slice = []
        width = []
        height = []
        for cls1list in cls1lists:
            cls1No.append(classes[0])
            allList.append(cls1list)
            cls1listDir = os.path.join(cls1Dir, cls1list)
            cls_no.append(0)
            data = nib.load(cls1listDir)
            img = data.get_fdata()
            #             print(type(img),img.shape)
            size.append(img.shape)
            width.append(img.shape[0])
            height.append(img.shape[1])
            n_slice.append(img.shape[2])

            cls1listsDir.append(cls1listDir)
        cls2listsDir = []
        for cls2list in cls2lists:
            cls_no.append(1)
            cls2No.append(classes[1])
            allList.append(cls2list)
            cls2listDir = os.path.join(cls2Dir, cls2list)
            data = nib.load(cls2listDir)
            img = data.get_fdata()
            #             print(type(img),img.shape)
            size.append(img.shape)
            width.append(img.shape[0])
            height.append(img.shape[1])
            n_slice.append(img.shape[2])
            cls2listsDir.append(os.path.join(cls2Dir, cls2list))
        cls1dict = zip(cls1No, cls1listsDir)
        cls2dict = zip(cls2No, cls2listsDir)
        df1 = pd.DataFrame(cls1dict, columns=['cls', 'dir'])
        df2 = pd.DataFrame(cls2dict, columns=['cls', 'dir'])
        df = pd.concat([df1, df2])
        df['name'] = allList
        df['size'] = size
        df['cls_no'] = cls_no
        df['width'] = width
        df['height'] = height
        df['n_slice'] = n_slice
        t = []
        for t_ in allList:
            t.append(t_.split('_')[1].split('.')[0])
            print("\r t is %s"%(t_),end="")


        df['t'] = t
        df = df.reset_index()
        print('dataframe is ok!')
        print('stratifiedKFold!')
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        for fold, (train_idx, val_idx) in enumerate(skf.split(df['dir'], df['cls'])):
            df.loc[val_idx, 'fold'] = fold
        # display(df.groupby(['fold', 'cls'])['name'].count())
        print('done!  csv file is in ',save_csv_dir)
        df.to_csv(save_csv_dir, index=None)
        return df


if __name__ == '__main__':
    # dir = '/mnt/llz/media/npcMri/cls/npc/'
    df = read_my_dir()
    # df.to_csv('/mnt/llz/media/npcMri/cls/cfgs/dataInfo/my.csv',index=None)

import pandas
import torch
import nibabel as nib
import numpy as np


def load_niigz(path,df):
    data=nib.load(path)
    img=data.get_fdata()
    img = img.astype('float32')
    cente_s=int(img.shape[2]/2)
    cente_w=int(img.shape[0]/2)
    cente_h=int(img.shape[1]/2)
    s_2=int(min(df['n_slice'])/2)
    w_2=int(min(df['width'])/2)
    h_2=int(min(df['height'])/2)
#     print(cente_w)
    img=img[cente_w-w_2:cente_w+w_2,cente_h-h_2:cente_h+h_2,cente_s-s_2:cente_s+s_2]
#     mx=np.max(img)
#     if mx:
#         img/=mx
    return img


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None):
        self.df = df
        self.label = label
        self.niigz_paths = df['dir'].tolist()
        self.clses = df['cls_no'].tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        niigz_path = self.niigz_paths[index]
        niigz = []
        niigz = load_niigz(niigz_path, self.df)
        if self.transforms:
            data = self.transforms(image=niigz)
            niigz = data['image']

        niigz = np.expand_dims(niigz, axis=3)
        niigz = np.transpose(niigz, (3, 2, 0,1))
        label=torch.tensor(self.clses[index])
        return torch.tensor(niigz),label


def myDataSet(df):
    mydataset=BuildDataset(df)
    return mydataset

def test():
    df=read_df('/mnt/llz/media/npcMri/cls/cfgs/dataInfo/my.csv')
    print('df:',df.shape)
    mydataset=myDataSet(df)
    n=0
    for _ in mydataset:
        n+=1
        if n==1:
            print(_.shape)
    print("共有 ",n," 个数据")

if __name__ == '__main__':
    test()



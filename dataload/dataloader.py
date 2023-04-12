
import json
from torch.utils.data import  DataLoader,Dataset

import sys
sys.path.append('/mnt/llz/code/cls/mymodel/dataload')
import dataset

sys.path.append('/mnt/llz/code/cls/mymodel/preprocess')
import get_csv

sys.path.append('/mnt/llz/code/cls/mymodel/utiles')
import get_json
import image_csv



def prepare_loaders(fold ,debug=False):
    datadict=get_json.open_json()
    # data_dir = datadict['data_dir']
    # csv_dir = datadict['csv_dir']
    # df=get_csv.read_my_dir(data_dir,csv_dir)
    batch_size=datadict['batch_size']
    num_workers=datadict['num_workers']
    df,df_test=image_csv.read_my_dir()
    # dataset.crop_img(df)
    # dataset.crop_img(df_test,is_train=False)
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)

    train_dataset = dataset.BuildDataset(train_df)
    valid_dataset = dataset.BuildDataset(valid_df)
    test_dataset=dataset.BuildDataset(df_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,drop_last=True,num_workers=num_workers)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
    return train_loader, valid_loader,test_loader

def test():
    train_loader,valid_loader,test_loader=prepare_loaders(fold=0)
    n = 0
    for x,y in test_loader:
        n += 1
        if n == 1:
            print(x.shape,y)
    print("共有 ", n, " 个数据")

if __name__ == '__main__':
    test()

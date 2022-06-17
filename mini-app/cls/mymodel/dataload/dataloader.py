
import json
from torch.utils.data import  DataLoader

import sys
sys.path.append('/mnt/llz/code/cls/mymodel/dataload')
import dataset

sys.path.append('/mnt/llz/code/cls/mymodel/preprocess')
import get_csv

sys.path.append('/mnt/llz/code/cls/mymodel/utiles')
import get_json





def prepare_loaders(fold ,debug=False):
    # datadict=get_json.open_json()
    # data_dir = datadict['data_dir']
    # csv_dir = datadict['csv_dir']
    # df=get_csv.read_my_dir(data_dir,csv_dir)
    df=get_csv.read_my_dir()
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    train_dataset = dataset.BuildDataset(train_df)
    valid_dataset = dataset.BuildDataset(valid_df)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=10, shuffle=False)
    return train_loader, valid_loader

def test():
    train_loader,valid_loader=prepare_loaders(fold=0)
    n = 0
    for x,y in train_loader:
        n += 1
        if n == 1:
            print(x.shape,y)
    print("共有 ", n, " 个数据")

if __name__ == '__main__':
    test()

import torchsummary
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import nibabel as nib
import pandas as pd
import numpy as np
import os
import logging
import json
import copy

import dataload.dataset as dataset
import dataload.dataloader as dataloader
import models.resnet3d_CBAM as resnet3d
import utiles.log as log

import sys
sys.path.append('/mnt/llz/code/cls/mymodel/utiles')
import get_json

mylog=log.MyLog()

get_json.json_data()


datadict=get_json.open_json()
num_epochs=datadict['num_epochs']
fold=datadict['fold']
resnet_layers=datadict['resnet-layers']
lr=datadict['lr']

# fold=0
# layers=10
# lr=1e-3
# csv_dir='/mnt/llz/media/npcMri/cls/cfgs/dataInfo/my.csv'



def main_3d():
    boardx_dir=mylog.boardx_dir
    info_dir = mylog.info_dir
    savemodel_dir=mylog.savemodel_dir

    model = resnet3d.generate_model(resnet_layers).cuda()
    mylog.write_info(torchsummary.summary(model, (1, 40, 224, 224)))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    with open(os.path.join(info_dir,'info.txt'),'w+') as f:
        f.write('lr is {}\n'.format(lr))
        f.write('fold is {}\n'.format(fold))
        f.write('layers is {}\n'.format(resnet_layers))
        f.write('epochs is {}\n'.format(num_epochs))



    train_loader, valid_loader = dataloader.prepare_loaders(fold=fold)




    print('tensorboardx_dir is ',boardx_dir)
    writer = SummaryWriter(log_dir=boardx_dir, flush_secs=60)
    writer.add_graph(model,torch.rand([1,1, 40, 224, 224]).cuda())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mylog.write_info('train on {}'.format(device))
    print('show process in \'tensorboard --logdir= xxx \' ')

    total_step = len(train_loader)
    time_list = []
    best_score=-np.inf
    best_epoch=-1
    best_model_wts=copy.deepcopy(model.state_dict())

    start=time.time()
    for epoch in range(num_epochs):
        start = time.time()
        per_epoch_loss = 0
        num_correct = 0
        val_num_correct = 0

        model.train()
        with torch.enable_grad():
            for x, label in tqdm(train_loader):
                x = x.to(device)
                label = label.to(device)
                label = torch.squeeze(label)  # label的形状是 [256,1] 要将其变成 [256]
                # Forward pass
                logits = model(x)
                loss = criterion(logits, label)

                per_epoch_loss += loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred = logits.argmax(dim=1)
                num_correct += torch.eq(pred, label).sum().float().item()
            print("Train Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}".format(epoch, per_epoch_loss / total_step,
                                                                        num_correct / len(train_loader.dataset)))
            mylog.write_info("Train Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}".format(epoch, per_epoch_loss / total_step,
                                                                        num_correct / len(train_loader.dataset)))
            writer.add_scalars('loss', {"loss": (per_epoch_loss / total_step)}, epoch)
            writer.add_scalars('acc', {"acc": num_correct / len(train_loader.dataset)}, epoch)

        model.eval()
        with torch.no_grad():
            for x, label in tqdm(valid_loader):
                x = x.to(device)
                label = label.to(device)
                label = torch.squeeze(label)
                # Forward pass
                logits = model(x)
                pred = logits.argmax(dim=1)
                val_num_correct += torch.eq(pred, label).sum().float().item()
            val_score=val_num_correct / len(valid_loader.dataset)
            print("val Epoch: {}\t Acc: {:.6f}".format(epoch, val_score))
            if val_score >=best_score:
                print(f"{epoch: }Valid Score Improved ({best_score:0.4f} ---> {val_score:0.4f})")
                best_score=val_score
                best_epoch=epoch
                PATH=os.path.join(savemodel_dir,'fold_'+str(fold)+'_epoch_'+str(epoch)+'.pt')
                torch.save(model.state_dict(),PATH)
                print(f"model saved!",PATH)
            mylog.write_info("val Epoch: {}\t Acc: {:.6f}".format(epoch, val_num_correct / len(valid_loader.dataset)))
            writer.add_scalars('acc', {"val_acc": val_num_correct / len(valid_loader.dataset)}, epoch)
            writer.add_scalars('time', {"time": (time.time() - start)}, epoch)


        scheduler.step()
    end=time.time()
    time_elapsed=end-start
    mylog.write_info('time is {}'.format(time_elapsed))
    mylog.write_info('best_epoch is {}'.format(best_epoch))
    mylog.write_info('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))


if __name__ == '__main__':
    main_3d()
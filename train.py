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
from models.pretrainedModel.pretrainmodels import get_resnet101,Net,get_convnext_base,get_densenet201,get_mobilenet_v3_large, get_vgg16
from models.mydesignModel.my_mobilenetV1 import MobileNet
import utiles.log as log
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" #此时显示4块显卡
import sys
sys.path.append('/mnt/llz/code/cls/mymodel/utiles')
import get_json
import test


get_json.json_data()


datadict=get_json.open_json()
num_epochs=datadict['num_epochs']
fold=datadict['fold']
# resnet_layers=datadict['resnet-layers']
lr=datadict['lr']
batch_size=datadict['batch_size']
mode=datadict['mode']
# fold=0
# layers=10
# lr=1e-3
# csv_dir='/mnt/llz/media/npcMri/cls/cfgs/dataInfo/my.csv'
mylog=log.MyLog(mode=mode)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True


def write_info(info,info_dir):
    with open(os.path.join(info_dir,'info.txt'),'a+') as f:
        f.write(info+'\n')
        f.write('\n')

def write_info_head(info,info_dir):
    with open(os.path.join(info_dir,'info.txt'),'a+') as f:
        f.write("-"*50+'\n')
        f.write("-" * 50+'\n')
        f.write("-----------------"+info+"------------------"+'\n')
        f.write("-" * 50+'\n')
        f.write("-" * 50+'\n')
        f.write('\n')

def main_2d():
    setup_seed(916)
    boardx_dir=mylog.boardx_dir
    info_dir = mylog.info_dir
    pred_dir=mylog.pred_dir
    savemodel_dir=mylog.savemodel_dir

    net,model_name = get_resnet101()
    # net, model_name = get_vgg16()
    # net,model_name = get_mobilenet_v3_large()


    # net, target = get_convnext_base()
    model=Net(net,model_name).cuda()
    # model = MobileNet().cuda()
    # model_name='my mobile net '

    mylog.write_info(torchsummary.summary(model, (3, 224, 224)))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    write_info_head("super param info", info_dir)
    write_info('lr is {}'.format(lr),info_dir)
    write_info('fold is {}'.format(fold), info_dir)
    write_info('epoches is {}'.format(num_epochs), info_dir)
    write_info('model_name is {}'.format(model_name), info_dir)



    train_loader, valid_loader,test_loader = dataloader.prepare_loaders(fold=fold)




    print('tensorboardx_dir is ',boardx_dir)
    writer = SummaryWriter(log_dir=boardx_dir, flush_secs=60)
    writer.add_graph(model,torch.rand([1,3, 224, 224]).cuda())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    write_info('train on {}'.format(device),info_dir)
    print('show process in \'tensorboard --logdir= xxx \' ')

    total_step = len(train_loader)
    best_score=-np.inf
    best_epoch=-1

    start=time.time()
    test_best_score = 0.0
    for epoch in range(num_epochs):
        start = time.time()
        per_epoch_loss = 0
        num_correct = 0
        val_num_correct = 0
        test_num_correct = 0

        model.train()
        with torch.enable_grad():
            for x, label in tqdm(train_loader):
                x = x.to(device)
                label = label.to(device)
                label = torch.squeeze(label)  # label的形状是 [256,1] 要将其变成 [256]
                # Forward pass
                logits = model(x)
                # print(logits.shape)
                # print(label.shape)
                loss = criterion(logits, label.long())

                per_epoch_loss += loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred = logits.argmax(dim=1)
                num_correct += torch.eq(pred, label).sum().float().item()
            # print("Train Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}".format(epoch, per_epoch_loss / total_step,
            #                                                             num_correct / len(train_loader.dataset)))

            mylog.write_info(
                    "Train Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}".format(epoch, per_epoch_loss / total_step,
                                                                          num_correct / len(train_loader.dataset)))
            writer.add_scalars('loss', {"loss": (per_epoch_loss / total_step)}, epoch)
            writer.add_scalars('acc', {"acc": num_correct / len(train_loader.dataset)}, epoch)

        model.eval()
        with torch.no_grad():
        # with torch.enable_grad():
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
                write_info_head("better epoch --- model saved", info_dir)
                print(f"  epoch {epoch }:  Valid Score Improved ({best_score:0.4f} ---> {val_score:0.4f})")
                write_info(f"  epoch {epoch}:  Valid Score Improved ({best_score:0.4f} ---> {val_score:0.4f})",info_dir)
                best_score=val_score
                best_epoch=epoch
                es=0
                PATH=os.path.join(savemodel_dir,'fold_'+str(fold)+'_epoch_'+str(epoch)+'_'+str(int(val_score*100))+'_val.pt')
                torch.save(model.state_dict(),PATH)
                write_info(f"model saved! path is "+str(PATH),info_dir)
                print(f"model saved! path is " + str(PATH))

                write_info("val Epoch: {}\t Acc: {:.6f}".format(epoch, val_num_correct / len(valid_loader.dataset)),info_dir)
            else:
                es+=1
                print("Counter {} of 6".format(es))
                if es>5:
                    print("Early stopping with best_acc: ",best_score)
                    break
            if epoch//10==0 or epoch//10==5:
                PATH=os.path.join(savemodel_dir,'fold_'+str(fold)+'_epoch_'+str(epoch)+'_'+str(int(val_score*100))+'_val.pt')
                torch.save(model.state_dict(), PATH)
                print(f"model saved! path is " + str(PATH))
            elif epoch>num_epochs-3:

                write_info_head(" last epoch --- model saved " , info_dir)
                write_info(str(epoch)+" model saved",info_dir)
                PATH=os.path.join(savemodel_dir,'fold_'+str(fold)+'_epoch_'+str(epoch)+'.pt')
                torch.save(model.state_dict(),PATH)
                print(f"model saved!",PATH)
                write_info("-" * 20,info_dir)
                write_info("last val Epoch: {}\t Acc: {:.6f}".format(epoch, val_num_correct / len(valid_loader.dataset)),info_dir)

                write_info("-" * 20,info_dir)
            writer.add_scalars('acc', {"val_acc": val_num_correct / len(valid_loader.dataset)}, epoch)
            writer.add_scalars('time', {"time": (time.time() - start)}, epoch)

        model.eval()
        with torch.no_grad():
            for x_t,label_t in tqdm(test_loader):
                x_t = x_t.to(device)
                label_t = label_t.to(device)
                label_t = torch.squeeze(label_t)
                # Forward pass
                logits_t = model(x_t)
                pred_t = logits_t.argmax(dim=1)
                test_num_correct += torch.eq(pred_t, label_t).sum().float().item()
            test_score = test_num_correct / len(test_loader.dataset)
            if test_score >=test_best_score:


                write_info_head("test better epoch --- model saved", info_dir)
                print(f"  epoch {epoch }:  Test Score Improved ({test_best_score:0.4f} ---> {test_score:0.4f})")
                write_info(f"  epoch {epoch}:  Test Score Improved ({test_best_score:0.4f} ---> {test_score:0.4f})",info_dir)
                test_best_score=test_score
                test_best_epoch=epoch
                PATH=os.path.join(savemodel_dir,'fold_'+str(fold)+'_epoch_'+str(epoch)+'_test_'+str(int(test_score*100))+'.pt')

                x = torch.randn(13,3, 224, 224, requires_grad=True).float().cuda()
                torch_out = model(x)
                torch.onnx.export(model, x, os.path.join(savemodel_dir,'fold_'+str(fold)+'_epoch_'+str(epoch)+'_test_'+str(int(test_score*100))+'.onnx'),
                                  export_params=True,verbose=True,input_names = ['input'],output_names = ['output'])

                torch.save(model.state_dict(),PATH)
                write_info(f"model saved! path is "+str(PATH),info_dir)
                print(f"model saved! path is " + str(PATH))
                test.calc(epoch,model,pred_dir)
                write_info("test Epoch: {}\t Acc: {:.6f}".format(epoch, test_num_correct / len(test_loader.dataset)),info_dir)
            writer.add_scalars('acc', {"test_acc": test_num_correct / len(test_loader.dataset)}, epoch)
            writer.add_scalars('time', {"time": (time.time() - start)}, epoch)
            print("test Epoch: {}\t Acc: {:.6f}".format(epoch, test_score))


        scheduler.step()
    end=time.time()
    time_elapsed=end-start

    write_info_head(" result infomation ", info_dir)

    write_info('time is {}'.format(time_elapsed),info_dir)
    write_info('best_epoch is {}'.format(best_epoch),info_dir)
    print('best_epoch is {}'.format(best_epoch))
    write_info('test_best_epoch is {}'.format(test_best_epoch),info_dir)
    print('test_best_epoch is {}'.format(test_best_epoch))
    write_info('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60),info_dir)

if __name__ == '__main__':
    main_2d()
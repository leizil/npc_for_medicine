import torch
import torch.nn as nn
import numpy as np
import os
from torch.nn import  CrossEntropyLoss
import torch.nn.functional  as F
from tqdm import tqdm
import dataload.dataset as dataset
import dataload.dataloader as dataloader
from models.pretrainedModel.pretrainmodels import get_resnet18,get_resnet101,Net,get_convnext_base
from models.mydesignModel.my_mobilenetV1 import MobileNet
import time
from torch.utils.tensorboard import SummaryWriter
import shutil

class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x,y):
        return torch.mean(torch.pow((x-y)),2)



def distill(teacher_dict_path):
    alpha=0.7
    device = 'cuda:0' if torch.cuda.is_available() else "cpu"
    num_epochs = 80
    train_loader, valid_loader, test_loader = dataloader.prepare_loaders(fold=0)

    best_score = -np.inf
    best_epoch = -1
    fold = 0

    loss_fun = CrossEntropyLoss()
    criterion = nn.KLDivLoss()



    net_teacher, model_name_teacher = get_resnet101()
    model_teacher = Net(net_teacher, model_name_teacher).cuda()
    net_pth = torch.load(
        teacher_dict_path, device)
    model_teacher.load_state_dict(net_pth)
    print('load teacher dict ! model name is ',model_name_teacher,)
    print('model path is ',teacher_dict_path)
    net_student, model_name_student = get_resnet18()
    model_student = Net(net_student, model_name_student).cuda()
    # model_student = MobileNet().cuda()
    # model_name_student='mobile net v1: '


    # date = time.strftime("%Y-%m-%d")
    date = teacher_dict_path.split("/")[4]
    print("date: ",date)
    distill_dir = os.path.join('/mnt/llz/log/', date, 'distill',"save_models")
    if not os.path.exists(distill_dir):
        os.mkdir(distill_dir)
    else:
        shutil.rmtree(distill_dir)
        os.mkdir(distill_dir)

    optimizer = torch.optim.Adam(model_student.parameters(), lr=0.01)
    total_step = len(train_loader)

    boardx_dir=os.path.join('/mnt/llz/log/', date, 'distill',"tensorboardx")
    if not os.path.exists(boardx_dir):
        os.mkdir(boardx_dir)
    else:
        shutil.rmtree(boardx_dir)
        os.mkdir(boardx_dir)
    writer = SummaryWriter(log_dir=boardx_dir, flush_secs=60)

    for epoch in range(num_epochs):
        per_epoch_loss = 0
        per_epoch_loss_hard=0
        per_epoch_loss_soft=0
        num_correct = 0
        val_num_correct = 0
        model_student.train()
        with torch.enable_grad():
            for x, label in tqdm(train_loader):
                x = x.to(device)
                label = label.to(device)
                label = torch.squeeze(label)  # label的形状是 [256,1] 要将其变成 [256]
                # Forward pass
                logits_teacher = model_teacher(x)
                logits_student = model_student(x)


                #损失函数
                T = 2
                outputs_S = F.log_softmax(logits_student / T, dim=1)
                outputs_T = F.softmax(logits_teacher / T, dim=1)

                loss_hard = loss_fun(logits_student, label.long())
                loss_soft = criterion(outputs_S, outputs_T) * T * T
                loss = alpha * loss_hard + (1-alpha) * loss_soft

                per_epoch_loss += loss.item()
                per_epoch_loss_hard += loss_hard.item()
                per_epoch_loss_soft += loss_soft.item()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred = logits_student.argmax(dim=1)
                num_correct += torch.eq(pred, label).sum().float().item()
            writer.add_scalars('loss', {"loss_all": (per_epoch_loss / total_step)}, epoch)
            writer.add_scalars('loss', {"loss_hard": (per_epoch_loss_hard / total_step)}, epoch)
            writer.add_scalars('loss', {"loss_soft": (per_epoch_loss_soft / total_step)}, epoch)
            writer.add_scalars('acc', {"acc": num_correct / len(train_loader.dataset)}, epoch)
            print(
                "Train Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}".format(epoch, per_epoch_loss / total_step,
                                                                      num_correct / len(train_loader.dataset)))
            # print('loss soft is ', loss_soft)
            # print('loss hard is ', loss_hard)
        model_student.eval()
        with torch.no_grad():
            for x, label in tqdm(valid_loader):
                x = x.to(device)
                label = label.to(device)
                label = torch.squeeze(label)
                # Forward pass
                logits = model_student(x)
                pred = logits.argmax(dim=1)

                val_num_correct += torch.eq(pred, label).sum().float().item()
            val_score = val_num_correct / len(valid_loader.dataset)
            writer.add_scalars('acc', {"val_acc": val_score}, epoch)
            print("val Epoch: {}\t Acc: {:.6f}".format(epoch, val_score))
            if val_score >= best_score:
                print(f"  epoch {epoch}:  Valid Score Improved ({best_score:0.4f} ---> {val_score:0.4f})")

                best_score = val_score
                best_epoch = epoch
                PATH = os.path.join(distill_dir, 'fold_' + str(fold) + '_student_epoch_' + str(epoch) + '_' + str(
                    int(val_score * 100)) + '.pt')
                torch.save(model_student.state_dict(), PATH)
                print(f"model saved! path is " + str(PATH))
                x = torch.randn(13,3, 224, 224, requires_grad=True).float().cuda()
                # torch_out = model(x)
                torch.onnx.export(model_student, x, os.path.join(distill_dir, 'fold_' + str(fold) + '_student_epoch_' + str(epoch) + '_' + str(
                    int(val_score * 100)) + '.onnx'),
                                  export_params=True,verbose=True,input_names = ['input'],output_names = ['output'])
            if epoch//10==0 or epoch//10==5:
                PATH = os.path.join(distill_dir, 'fold_' + str(fold) + '_student_epoch_' + str(epoch) + '_' + str(
                    int(val_score * 100)) + '.pt')
                torch.save(model_student.state_dict(), PATH)
                print(f"model saved! path is " + str(PATH))

if __name__ == '__main__':
    path='/mnt/llz/log/2023-03-27/save_models/fold_0_epoch_1_test_99.pt'
    distill(teacher_dict_path=path)

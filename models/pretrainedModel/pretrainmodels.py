import torchvision
import torch
import torchinfo
import torch.nn as nn
import torchsummary
import numpy as np
import sys
sys.path.append('/mnt/llz/code/cls/mymodel/models/pretrainedModel')
import  convnext
import mobilenet.mobilenet_v3 as mobilenet_v3
# sys.path.append('/mnt/llz/code/cls/mymodel/models/torchvision113')
# import models.convnext as torchvision113_models_convnext

models_in_ch={'resnet18':512,'vgg16':512,'alexnet':256,'resnet101':2048,
              'resnet152':2048,'densenet121':512,'densenet161':512,
              'densenet169':512,'densenet201':2048,'mobilenet_v2':512,
              'mobilenet_v3_large':960,'mobilenet_v3_small':576,
              'convnext_base':1024,'convnext_large':1536,'convnext_small':768}

def get_pretrained_dict():
    model_urls = {
        # 'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
        # 'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
        # 'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
        # 'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        # 'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
        # 'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
        # 'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
        # 'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
        # 'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
        # 'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        # 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        # 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        # 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        # 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        # 'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
        # 'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
        # 'resnext101_32x16d': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
        # 'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
        # 'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
        # 'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
        # 'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
        # 'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
        # 'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
        # 'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
        # 'mobilenet_v3_small': "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
        # 'mobilenet_v3_large': "https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth",
        # "mnasnet0_5": "https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth",
        # "mnasnet0_75": None,
        # "mnasnet1_0": "https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth",
        # "mnasnet1_3": None,
        "convnext_tiny": "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
        "convnext_small": "https://download.pytorch.org/models/convnext_small-0c510722.pth",
        "convnext_base": "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
        "convnext_large": "https://download.pytorch.org/models/convnext_large-ea097f82.pth",
        # "regnet_y_400mf": "https://download.pytorch.org/models/regnet_y_400mf-e6988f5f.pth",
        # "regnet_y_800mf": "https://download.pytorch.org/models/regnet_y_800mf-58fc7688.pth",
        # "regnet_y_1_6gf": "https://download.pytorch.org/models/regnet_y_1_6gf-0d7bc02a.pth",
        # "regnet_y_3_2gf": "https://download.pytorch.org/models/regnet_y_3_2gf-9180c971.pth",
        # "regnet_y_8gf": "https://download.pytorch.org/models/regnet_y_8gf-dc2b1b54.pth",
        # "regnet_y_16gf": "https://download.pytorch.org/models/regnet_y_16gf-3e4a00f9.pth",
        # "regnet_y_32gf": "https://download.pytorch.org/models/regnet_y_32gf-8db6d4b5.pth",
        # "regnet_y_128gf": "https://download.pytorch.org/models/regnet_y_128gf_swag-c8ce3e52.pth",
        # "regnet_x_400mf": "https://download.pytorch.org/models/regnet_x_400mf-62229a5f.pth",
        # "regnet_x_800mf": "https://download.pytorch.org/models/regnet_x_800mf-94a99ebd.pth",
        # "regnet_x_1_6gf": "https://download.pytorch.org/models/regnet_x_1_6gf-a12f2b72.pth",
        # "regnet_x_3_2gf": "https://download.pytorch.org/models/regnet_x_3_2gf-7071aa85.pth",
        # "regnet_x_8gf": "https://download.pytorch.org/models/regnet_x_8gf-2b70d774.pth",
        # "regnet_x_16gf": "https://download.pytorch.org/models/regnet_x_16gf-ba3796d7.pth",
        # "regnet_x_32gf": "https://download.pytorch.org/models/regnet_x_32gf-6eb8fdc6.pth",
    }
    for model in model_urls:
        print('-' * 100)
        print(model)
        if model_urls[model] == None:
            continue
        torch.utils.model_zoo.load_url(model_urls[model], model_dir='./pretrained_pth_file')
        print('-' * 100)

class Classifer(nn.Module):
    def __init__(self, in_ch, num_classes):
        super(Classifer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        # print('!!! classier head ',x.shape)
        x = torch.flatten(x, 1)
        # print('!!! classier head flatten shape', x.shape)
        x = self.fc(x)
        # print('!!! classier head out shape', x.shape)
        return x

def get_alexnet(pretrained=False):
    model = torchvision.models.alexnet(pretrained=pretrained)
    pretrained_pth = torch.load(
        '/mnt/llz/code/cls/mymodel/models/pretrainedModel/pretrained_pth_file/alexnet-owt-4df8aa71.pth')
    model.load_state_dict(pretrained_pth)
    name='alexnet'
    return model,name


def get_vgg16(pretrained=False):
    model = torchvision.models.vgg16(pretrained=pretrained)
    pretrained_pth = torch.load(
        '/mnt/llz/code/cls/mymodel/models/pretrainedModel/pretrained_pth_file/vgg16-397923af.pth')
    model.load_state_dict(pretrained_pth)
    name='vgg16'
    return model,name

def get_resnet101(pretrained=False):
    model = torchvision.models.resnet101(pretrained=pretrained)
    pretrained_pth = torch.load(
        '/mnt/llz/code/cls/mymodel/models/pretrainedModel/pretrained_pth_file/resnet101-5d3b4d8f.pth')
    model.load_state_dict(pretrained_pth)
    name='resnet101'
    return model,name


def get_resnet18(pretrained=False):
    model = torchvision.models.resnet18(pretrained=pretrained)
    pretrained_pth = torch.load(
        '/mnt/llz/code/cls/mymodel/models/pretrainedModel/pretrained_pth_file/resnet18-5c106cde.pth')
    model.load_state_dict(pretrained_pth)
    target='resnet18'
    return model,target


def get_resnet152(pretrained=False):
    model = torchvision.models.resnet152(pretrained=pretrained)
    pretrained_pth = torch.load(
        '/mnt/llz/code/cls/mymodel/models/pretrainedModel/pretrained_pth_file/resnet152-b121ed2d.pth')
    model.load_state_dict(pretrained_pth)
    target='resnet152'
    return model,target


def get_densenet121(pretrained=False):
    model = torchvision.models.densenet121(pretrained=pretrained)
    pretrained_pth = torch.load(
        '/mnt/llz/code/cls/mymodel/models/pretrainedModel/pretrained_pth_file/densenet121-a639ec97.pth')
    model.load_state_dict(pretrained_pth)
    target='densenet121'
    return model,target

def get_densenet161(pretrained=False):
    model = torchvision.models.densenet161(pretrained=pretrained)
    pretrained_pth = torch.load(
        '/mnt/llz/code/cls/mymodel/models/pretrainedModel/pretrained_pth_file/densenet161-8d451a50.pth')
    model.load_state_dict(pretrained_pth)
    target='densenet161'
    return model,target

def get_densenet169(pretrained=False):
    model = torchvision.models.densenet169(pretrained=pretrained)
    pretrained_pth = torch.load(
        '/mnt/llz/code/cls/mymodel/models/pretrainedModel/pretrained_pth_file/densenet169-b2777c0a.pth')
    model.load_state_dict(pretrained_pth)
    target='densenet169'
    return model,target

from torch import Tensor

def get_densenet201(pretrained=False):
    model = torchvision.models.densenet201(pretrained=True)
    if pretrained:
        pretrained_pth = torch.load(
            '/mnt/llz/code/cls/mymodel/models/pretrainedModel/pretrained_pth_file/densenet201-c1103571.pth')
        model.load_state_dict(pretrained_pth)
    target='densenet201'
    model.cpu()
    torchsummary.summary(model, (3, 224, 224),device='cpu')
    return model,target

def get_mobilenet_v2(pretrained=False):
    model = torchvision.models.mobilenet_v2(pretrained=pretrained)
    pretrained_pth = torch.load(
        '/mnt/llz/code/cls/mymodel/models/pretrainedModel/pretrained_pth_file/mobilenet_v2-b0353104.pth')
    model.load_state_dict(pretrained_pth)
    target='mobilenet_v2'
    return model,target

def get_mobilenet_v3_large(pretrained=False):
    model = mobilenet_v3.mobilenet_v3_large()
    pretrained_pth = torch.load(
        '/mnt/llz/code/cls/mymodel/models/pretrainedModel/pretrained_pth_file/mobilenet_v3_large-5c1a4163.pth')
    model.load_state_dict(pretrained_pth)
    target='mobilenet_v3_large'
    return model,target

def get_mobilenet_v3_small(pretrained=False):
    model = mobilenet_v3.mobilenet_v3_small()
    pretrained_pth = torch.load(
        '/mnt/llz/code/cls/mymodel/models/pretrainedModel/pretrained_pth_file/mobilenet_v3_small-047dcff4.pth')
    model.load_state_dict(pretrained_pth)
    target='mobilenet_v3_small'
    return model,target

def get_convnext_base(pretrained=False):
    model = convnext.convnext_base()
    pretrained_pth = torch.load(
        '/mnt/llz/code/cls/mymodel/models/pretrainedModel/pretrained_pth_file/convnext_base-6075fbad.pth')
    model.load_state_dict(pretrained_pth)
    target='convnext_base'
    return model,target

def get_convnext_large(pretrained=True):
    model = convnext.convnext_large()
    if pretrained:
        pretrained_pth = torch.load(
            '/mnt/llz/code/cls/mymodel/models/pretrainedModel/pretrained_pth_file/convnext_large-ea097f82.pth')
        model.load_state_dict(pretrained_pth)
    target='convnext_large'
    return model,target

def get_convnext_small(pretrained=False):
    model =  convnext.convnext_small()
    pretrained_pth = torch.load(
        '/mnt/llz/code/cls/mymodel/models/pretrainedModel/pretrained_pth_file/convnext_small-0c510722.pth')
    model.load_state_dict(pretrained_pth)
    target='convnext_small'
    return model,target


class Net(nn.Module):
    def __init__(self, model,model_name):
        super(Net, self).__init__()
        self.backbone = nn.Sequential(*list(model.children())[:-3])
        # print(list(model.children()))
        # self.transion_layer=nn.ConvTranspose2d(2048,2048,kernel_size=4,stride=3)
        # self.pool_layer=nn.MaxPool2d(32)
        # self.Linear_layer = nn.Linear(1000, 2)
        classifer=Classifer(models_in_ch[model_name],2)
        if model_name=='densenet201':
            self.classification_head1 = nn.Sequential(*list(model.children())[-2],
                                                  classifer)
        else:
            self.classification_head1 = nn.Sequential(*list(model.children())[-3],
                                                  classifer)
        # self.classification_head2 = nn.Sequential(*list(model.children())[-3],
        #                                           classifier(256, 5))

    def forward(self, x):
        x = self.backbone(x)
        # print("!!!!!!!!", x.shape)
        output1 = self.classification_head1(x)
        return output1


if __name__ == '__main__':
    # get_pretrained_dict()
    # backbone,name=get_alexnet()
    # backbone, name = get_resnet101()
    # backbone, name = get_resnet152()
    # backbone, name = get_convnext_large()
    # backbone, name = get_convnext_large()
    # backbone,name=get_mobilenet_v2()
    # backbone, name = get_mobilenet_v3_large()
    # backbone, name = get_mobilenet_v3_small()
    backbone,name = get_vgg16()
    # backbone,name = get_resnet18()
    # backbone, name = get_densenet201()
    backbone.cuda()
    torchsummary.summary(backbone, (3, 224, 224))
    for i in backbone.children():
        print(i)
    backbone.cuda()
    mynet = Net(model=backbone,model_name=name).cuda()
    # torchinfo.summary(backbone, (1, 3, 224, 224))

    # print('-' * 20)
    torchsummary.summary(mynet, (3, 224, 224))

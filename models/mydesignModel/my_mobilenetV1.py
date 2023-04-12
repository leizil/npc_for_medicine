import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import torchsummary
import numpy as np

class ConvBNReLU(nn.Sequential):
    def __init__(self,in_planes:int,out_planes:int,kernel_size:int=3,stride:int =1,groups:int =1,dilation:int=1):
        padding=(kernel_size-1)
        super(ConvBNReLU,self).__init__(
            nn.Conv2d(in_planes,out_planes,kernel_size,stride,padding,groups=groups,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.out_channels =out_planes

class DWConvBNReLU(nn.Sequential):
    def __init__(self,in_planes:int,out_planes:int,kernel_size:int=3,stride:int =1,groups:int =3,dilation:int=1):
        padding=(kernel_size-1)//2
        super(DWConvBNReLU,self).__init__(
            nn.Conv2d(in_planes,out_planes,kernel_size,stride,padding,groups=groups,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.out_channels =out_planes


class PWConvBNReLU(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int = 1, stride: int = 1, groups: int = 1):
        padding = (kernel_size - 1) // 2
        super(PWConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.out_channels = out_planes


PWConv1x1BNReLU = PWConvBNReLU
Conv3x3BNReLU=ConvBNReLU
DWConv3x3BNReLU=DWConvBNReLU

class MobileNet(nn.Module):
    def __init__(self,num_classes=2,width_factor=2,):
        super(MobileNet,self).__init__()
        self.Conv1=Conv3x3BNReLU(in_planes=3,out_planes=32,stride=2,groups=1)
        self.DWConv2=DWConv3x3BNReLU(in_planes=32,out_planes=32,stride=1,groups=32)
        self.PWConv3=PWConv1x1BNReLU(in_planes=32,out_planes=64,stride=1,groups=1)
        self.DWConv4=DWConv3x3BNReLU(in_planes=64,out_planes=64,stride=2,groups=64)
        self.PWConv5=PWConv1x1BNReLU(in_planes=64,out_planes=128,stride=1,groups=1)
        self.DWConv6=DWConv3x3BNReLU(in_planes=128,out_planes=128,stride=1,groups=128)
        self.PWConv7=PWConv1x1BNReLU(in_planes=128,out_planes=128,stride=1,groups=1)
        self.DWConv8 = DWConv3x3BNReLU(in_planes=128,out_planes=128,stride=2,groups=128)
        # Conv / s1     1 × 1 × 128 × 256   28 × 28 × 128    逐点卷积，步长为1，特征图尺寸减半
        self.PWConv9 = PWConv1x1BNReLU(in_planes=128,out_planes=256,stride=1,groups=1)
        # Conv dw / s1  3 × 3 × 256 dw      28 × 28 × 256    深度卷积，步长为1，特征图尺寸不变
        self.DWConv10 = DWConv3x3BNReLU(in_planes=256,out_planes=256,stride=1,groups=256)
        # Conv / s1     1 × 1 × 256 × 256   28 × 28 × 256
        self.PWConv11 = PWConv1x1BNReLU(in_planes=256,out_planes=256,stride=1,groups=1)
        # Conv dw / s2  3 × 3 × 256 dw      28 × 28 × 256
        self.DWConv12 = DWConv3x3BNReLU(in_planes=256,out_planes=256,stride=1,groups=256)
        # Conv / s1     1 × 1 × 256 × 512   14 × 14 × 256
        self.PWConv13 = PWConvBNReLU(in_planes=256,out_planes=512,kernel_size=3,stride=1,groups=1)
        # Conv dw / s1  3 × 3 × 512 dw      14 × 14 × 512   Conv / s1   1 × 1 × 512 × 512  14 × 14 × 512  x5
        self.DWConv14 = DWConv3x3BNReLU(in_planes=512,out_planes=512,stride=1,groups=512)
        self.PWConv15 = PWConv1x1BNReLU(in_planes=512,out_planes=512,stride=1,groups=1)
        self.DWConv16 = DWConv3x3BNReLU(in_planes=512,out_planes=512,stride=1,groups=512)
        self.PWConv17 = PWConv1x1BNReLU(in_planes=512,out_planes=512,stride=1,groups=1)
        self.DWConv18 = DWConv3x3BNReLU(in_planes=512,out_planes=512,stride=1,groups=512)
        self.PWConv19 = PWConv1x1BNReLU(in_planes=512,out_planes=512,stride=1,groups=1)
        self.DWConv20 = DWConv3x3BNReLU(in_planes=512,out_planes=512,stride=1,groups=512)
        self.PWConv21 = PWConv1x1BNReLU(in_planes=512,out_planes=512,stride=1,groups=1)
        self.DWConv22 = DWConv3x3BNReLU(in_planes=512,out_planes=512,stride=1,groups=512)
        self.PWConv23 = PWConv1x1BNReLU(in_planes=512,out_planes=512,stride=1,groups=1)
        # Conv dw / s2  3 × 3 × 512 dw     14 × 14 × 512
        self.DWConv24 = DWConv3x3BNReLU(in_planes=512,out_planes=512,stride=2,groups=512)
        # Conv / s1     1 × 1 × 512 × 1024 7 × 7 × 512
        self.PWConv25 = PWConvBNReLU(in_planes=512,out_planes=1024,kernel_size=3,stride=1,groups=1)
        # Conv dw / s2  3 × 3 × 1024 dw    7 × 7 × 1024
        self.DWConv26 = DWConv3x3BNReLU(in_planes=1024,out_planes=1024,stride=1,groups=1024)
        # Conv / s1     1 × 1 × 1024 × 1024 7 × 7 × 1024
        self.PWConv27 = PWConvBNReLU(in_planes=1024,out_planes=1024,stride=1,kernel_size=3,groups=1)
        # Avg Pool / s1 Pool 7 × 7 7 × 7 × 1024
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=1024,out_features=num_classes)
        self.init_param()
    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)
    def forward(self,x):
        x = self.Conv1(x)
        x = self.DWConv2(x)
        x = self.PWConv3(x)
        x = self.DWConv4(x)
        x = self.PWConv5(x)
        x = self.DWConv6(x)
        x = self.PWConv7(x)
        x = self.DWConv8(x)
        x = self.PWConv9(x)
        x = self.DWConv10(x)
        x = self.PWConv11(x)
        x = self.DWConv12(x)
        x = self.PWConv13(x)
        x = self.DWConv14(x)
        x = self.PWConv15(x)
        x = self.DWConv16(x)
        x = self.PWConv17(x)
        x = self.DWConv18(x)
        x = self.PWConv19(x)
        x = self.DWConv20(x)
        x = self.PWConv21(x)
        x = self.DWConv22(x)
        x = self.PWConv23(x)
        x = self.DWConv24(x)
        x = self.PWConv25(x)
        x = self.DWConv26(x)
        x = self.PWConv27(x)
        # print('before avg: shape is ',x.shape)
        x = self.avgpool(x)
        # print('after avg shape is ',x.shape)
        x = x.view(-1,1024)
        # print('after view shape is ', x.shape)
        x = self.dropout(x)
        x = self.fc(x)
        return x



def test():
    mobilenet=MobileNet(num_classes=2).cuda()
    torchsummary.summary(mobilenet,(3,224,224))


if __name__ == '__main__':
    test()
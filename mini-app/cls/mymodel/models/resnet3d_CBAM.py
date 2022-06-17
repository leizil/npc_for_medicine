import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.functional as F
import torchsummary


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv3d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #         avg_out=self.fc2(self.relu1(self.fc1(np.transpose(self.avg_pool(x).detach(),(0,2,1,3,4)))))
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # print('avg_out shape  is  ', avg_out.shape)
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # print('max_out shape  is  ', max_out.shape)
        out = avg_out + max_out
        # print('out shape  is  ', out.shape)
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print('avg_out.shape is ', avg_out.shape)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print('max_out.shape is ', max_out.shape)
        x = torch.cat([avg_out, max_out], dim=1)
        # print('x.shape is ', x.shape)
        x = self.conv1(x)
        return self.sigmoid(x)


# sa = SpatialAttention()
# torchsummary.summary(sa, (1, 40, 224, 224))


# ca = ChannelAttenstion(in_planes=40)
# torchsummary.summary(ca, (1, 40, 224, 224))

def get_channels():
    return [64,128,256,512]

def conv3x3x3(in_channels,out_channels,stride=1):
    return nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )
def conv1x1x1(in_channels,out_channels,stride=1):
    return nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,bias=False
    )

class BasicBlock(nn.Module):
    expansion=1
    def __init__(self,in_channels,channels,stride=1,downsample=None):
        super().__init__()

        self.conv1=conv1x1x1(in_channels,channels)
        self.bn1=nn.BatchNorm3d(channels)
        self.conv2=conv3x3x3(channels,channels,stride=stride)
        self.bn2=nn.BatchNorm3d(channels)
        self.conv3=conv1x1x1(channels,channels*self.expansion)
        self.bn3=nn.BatchNorm3d(channels*self.expansion)
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        residual=x

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out+=residual
        out=self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion=4

    def __init__(self,in_channels,channels,stride=1,downsample=None):
        super().__init__()

        self.conv1=conv1x1x1(in_channels,channels)
        self.bn1=nn.BatchNorm3d(channels)

        self.conv2=conv3x3x3(channels,channels,stride=stride)
        self.bn2=nn.BatchNorm3d(channels)

        self.conv3=conv1x1x1(channels,channels*self.expansion)
        self.bn3=nn.BatchNorm3d(channels*self.expansion)
        self.relu=nn.ReLU(inplace=True)

        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        residual=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)

        if self.downsample is not None:
            residual=self.downsample(x)

        out +=residual
        out=self.relu(out)
        return out

class ResNet3d(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=2):
        super().__init__()

        block_inplanes=[int(x*widen_factor) for x in block_inplanes]
        self.in_planes=block_inplanes[0]
        self.no_max_pool=no_max_pool

        self.ca=ChannelAttention(self.in_planes)
        self.sa=SpatialAttention()

        self.conv1=nn.Conv3d(n_input_channels,self.in_planes,
                             kernel_size=(conv1_t_size,7,7),
                             stride=(conv1_t_stride,2,2),
                             padding=(conv1_t_size//2,3,3),
                             bias=False)
        self.bn1=nn.BatchNorm3d(self.in_planes)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool3d(kernel_size=3,stride=2,padding=1)
        self.layer1=self._make_layer(block,block_inplanes[0],layers[0],
                                     shortcut_type)
        self.layer2=self._make_layer(block,block_inplanes[1],
                                     layers[1],shortcut_type,stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(block_inplanes[3] * block.expansion, 400)
        self.fc2=nn.Linear(400,n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out
    def _make_layer(self,block,planes,blocks,shortcut_type,stride=1):
        downsample=None
        if stride !=1 or self.in_planes!=planes*block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample=nn.Sequential(
                    conv1x1x1(self.in_planes,planes*block.expansion,stride=stride),
                    nn.BatchNorm3d(planes*block.expansion)
                )
        layers = []
        layers.append(block(in_channels=self.in_planes,
                            channels=planes,
                            stride=stride,
                            downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1,blocks):
            layers.append(block(self.in_planes,planes))

        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x=self.ca(x)*x
        x=self.sa(x)*x

        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet3d(BasicBlock, [1, 1, 1, 1], get_channels(), **kwargs)
    elif model_depth == 18:
        model = ResNet3d(BasicBlock, [2, 2, 2, 2], get_channels(), **kwargs)
    elif model_depth == 34:
        model = ResNet3d(BasicBlock, [3, 4, 6, 3], get_channels(), **kwargs)
    elif model_depth == 50:
        model = ResNet3d(Bottleneck, [3, 4, 6, 3], get_channels(), **kwargs)
    elif model_depth == 101:
        model = ResNet3d(Bottleneck, [3, 4, 23, 3], get_channels(), **kwargs)
    elif model_depth == 152:
        model = ResNet3d(Bottleneck, [3, 8, 36, 3], get_channels(), **kwargs)
    elif model_depth == 200:
        model = ResNet3d(Bottleneck, [3, 24, 36, 3], get_channels(), **kwargs)

    return model

def test():
    # mynet=BasicBlock(1,32)
    # mynet = Bottleneck(1, 32)
    mynet = generate_model(101).cuda()
    torchsummary.summary(mynet,(1,40,224,224))


if __name__ == '__main__':
    test()
    # pass


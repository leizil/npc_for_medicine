# 根据torch官方代码修改的convnext代码
# 模型下载地址：
#           tiny --- https://download.pytorch.org/models/convnext_tiny-983f1562.pth
#          small --- https://download.pytorch.org/models/convnext_small-0c510722.pth
#           base --- https://download.pytorch.org/models/convnext_base-6075fbad.pth
#          large --- https://download.pytorch.org/models/convnext_large-ea097f82.pth

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from typing import Any, Callable, List, Optional, Sequence
from functools import partial

# 定义了当你使用 from <module> import * 导入某个模块的时候能导出的符号
__all__ = [
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
]


# 定义随机深度层，简单说就是随机使得一个张量变为全0的张量
class StochasticDepth(nn.Module):

    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    # 定义随机深度变换函数的核心函数，简单说就是随机让输入张量变为全0，在外层组合一个类似resnet的短接就实现了随机深度变换函数
    # 当输入张量变为全0时，等效于将本层剔除，实现了n-1层的输出直接送入n+1层，相当于将第n层屏蔽掉了，本函数可实现两种屏蔽模式
    # 第一种是batch模式，就是一个batch内所有样本统一使用同一个随机屏蔽系数，第二种是row模式，就是一个batch内每个样本都有自己的系数
    def stochastic_depth(self, input: Tensor, p: float, mode: str, training: bool = True) -> Tensor:

        if p < 0.0 or p > 1.0:
            raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
        if mode not in ["batch", "row"]:
            raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
        if not training or p == 0.0:
            return input

        survival_rate = 1.0 - p
        if mode == "row":
            size = [input.shape[0]] + [1] * (input.ndim - 1)
        else:
            size = [1] * input.ndim
        noise = torch.empty(size, dtype=input.dtype, device=input.device)  # 基于所选模式，定义随机参数
        noise = noise.bernoulli_(survival_rate)  # 按照概率生成随机参数
        if survival_rate > 0.0:  # 概率为0的不需要做任何操作，但是概率为1的需要除以生存概率，这个类似于dropout
            noise.div_(survival_rate)  # 需要除以生存概率，以保证在平均值统计上，加不加随机深度层时是相同的
        return input * noise

    def forward(self, input: Tensor) -> Tensor:
        return self.stochastic_depth(input, self.p, self.mode, self.training)


# 定义一个卷积+归一化+激活函数层，这个类仅仅在convnext的stem层使用了一次
# torch.nn.Sequential相当于tf2.0中的keras.Sequential()，其实就是以最简单的方式搭建序列模型，不需要写forward()函数，
# 直接以列表形式将每个子模块送进来就可以了，或者也可以使用OrderedDict()或add_module()的形式向模块中添加子模块
# https://blog.csdn.net/weixin_42486623/article/details/122822580
class Conv2dNormActivation(torch.nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional[int] = None,
            groups: int = 1,
            norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            dilation: int = 1,
            inplace: Optional[bool] = True,
            bias: Optional[bool] = None
    ) -> None:

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        layers = [
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))

        super().__init__(*layers)  # 直接以列表的形式向torch.nn.Sequential中添加子模块


# 定义一个LN类，仅仅在模型的stem层、下采样层和分类层使用了该类，在基本模块中使用的是原生的nn.LayerNorm
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)  # 将通道维移到最后，对每一个像素的所有维度进行归一化，而非对所有像素的所有维度进行归一化
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)  # 将通道维移到第二位，后面使用时仅仅送入一个参数，所以是对最后一维进行归一化
        return x


# 定义通道转换类，在基本模块中被使用
class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.Tensor.permute(x, self.dims)


# 定义convnext的最基本模块，包含7*7卷积 + LN + 1*1卷积 + GELU + 1*1卷积 + 层缩放 + 随机深度
class CNBlock(nn.Module):
    def __init__(
            self,
            dim,
            layer_scale: float,
            stochastic_depth_prob: float,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:  # 实际使用时设置为None，所以基本模块使用的都是nn.LayerNorm
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(  # 用nn.Sequential搭建一个子模块，不需要重写forward()函数
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),  # 深度可分离卷积+反残差结构+替换和减少LN和激活函数
            Permute([0, 2, 3, 1]),  # 实现基本模块的方式有两种，分别是7*7卷积 + Permute + LN + Permute + 1*1卷积 + GELU + 1*1卷积
            norm_layer(dim),  # 或者7*7卷积 + Permute + LN + Linear + GELU + Linear + Permute
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),  # 经过验证第二种方式运行速度更快，所以本代码使用第二种方式
            nn.GELU(),  # 这里需要强调的是，本代码中的LN并不是标准LN，其并非对一个样本中所有通道的所有像素进行归一化，而是对所有
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),  # 通道的每个像素分别进行归一化，在torch中并没有直接实现该
            Permute([0, 3, 1, 2]),  # 变换的函数，所以只能通过Permute + LN + Permute组合的方式实现
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")  # 一个batch中不同样本使用不同的随机数

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)  # 关于layer_scale有个疑问，默认的值为何设置的非常小，而不是1
        result = self.stochastic_depth(result)
        result += input  # 短接，短接与stochastic_depth配合才能实现随机深度的思想
        return result


# 定义整个convnext的每个大模块的配置信息，整个模型的bottleneck层由四个大模块组成，每个大模块又包含很多基本模块
class CNBlockConfig:
    def __init__(self, input_channels: int, out_channels: Optional[int], num_layers: int) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers


# 根据配置列表搭建整个convnext模型
class ConvNeXt(nn.Module):
    def __init__(
            self,
            block_setting: List[CNBlockConfig],  # 参数配置列表，实际使用时，包含4个CNBlockConfig
            stochastic_depth_prob: float = 0.0,
            layer_scale: float = 1e-6,
            num_classes: int = 1000,
            block: Optional[Callable[..., nn.Module]] = None,  # 实际使用时，采用默认的None
            norm_layer: Optional[Callable[..., nn.Module]] = None,  # 实际使用时，采用默认的None
            **kwargs: Any,  # 实际使用时，没有其它参数输入
    ) -> None:
        super().__init__()

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:  # 所以实际使用的就是CNBlock
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)  # 所以实际使用的就是LayerNorm2d，仅用于模型的stem层、下采样层和分类层

        layers: List[nn.Module] = []

        ### 0. 搭建整个模型的第一层，即Stem层，包含卷积+偏置+LN
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=4,  # stride和kernel_size均为4
                stride=4,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        )
        ### 1. 搭建整个模型的第二部分，即bottleneck层
        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)  # 统计总的基本模块数量，用于计算随机深度的概率值
        stage_block_id = 0  # 这个概率值是越往后面越大，即浅层时尽量不要改变深度
        for cnf in block_setting:  # 遍历四个大模块配置，调整了大模块中小模块的比例
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):  # 遍历每个大模块中的基本模块
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))  # 用nn.Sequential搭建一个子模块，不需要重写forward()函数
            if cnf.out_channels is not None:  # 定义下采样层，前三个大模块结束后各使用了一次
                layers.append(
                    nn.Sequential(  # 用nn.Sequential搭建一个子模块，不需要重写forward()函数
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )
        self.features = nn.Sequential(*layers)  # 用nn.Sequential搭建一个子模块，不需要重写forward()函数
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        ### 2. 搭建最后的分类层
        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
        )
        self.classifier = nn.Sequential(  # 用nn.Sequential搭建一个子模块，不需要重写forward()函数
            norm_layer(lastconv_output_channels), nn.Flatten(1), nn.Linear(lastconv_output_channels, num_classes)
        )
        # 初始化参数
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


##############################################################################################################################
## 通过修改配置列表实现不同模型的定义
def convnext_tiny(num_classes: int = 1000, layer_scale: float = 1e-6) -> ConvNeXt:
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = 0.1
    return ConvNeXt(block_setting, stochastic_depth_prob, layer_scale, num_classes)


def convnext_small(num_classes: int = 1000, layer_scale: float = 1e-6) -> ConvNeXt:
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 27),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = 0.4
    return ConvNeXt(block_setting, stochastic_depth_prob, layer_scale, num_classes)


def convnext_base(num_classes: int = 1000, layer_scale: float = 1e-6) -> ConvNeXt:
    block_setting = [
        CNBlockConfig(128, 256, 3),
        CNBlockConfig(256, 512, 3),
        CNBlockConfig(512, 1024, 27),
        CNBlockConfig(1024, None, 3),
    ]
    stochastic_depth_prob = 0.5
    return ConvNeXt(block_setting, stochastic_depth_prob, layer_scale, num_classes)


def convnext_large(num_classes: int = 1000, layer_scale: float = 1e-6) -> ConvNeXt:
    block_setting = [
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 3),
        CNBlockConfig(768, 1536, 27),
        CNBlockConfig(1536, None, 3),
    ]
    stochastic_depth_prob = 0.5
    return ConvNeXt(block_setting, stochastic_depth_prob, layer_scale, num_classes)


if __name__ == "__main__":
    import torchvision.transforms as transforms
    from PIL import Image


    # 等比例拉伸图片，多余部分填充value
    def resize_padding(image, target_length, value=0):
        h, w = image.size  # 获得原始尺寸
        ih, iw = target_length, target_length  # 获得目标尺寸
        scale = min(iw / w, ih / h)  # 实际拉伸比例
        nw, nh = int(scale * w), int(scale * h)  # 实际拉伸后的尺寸
        image_resized = image.resize((nh, nw), Image.ANTIALIAS)  # 实际拉伸图片
        image_paded = Image.new("RGB", (ih, iw), value)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_paded.paste(image_resized, (dh, dw, nh + dh, nw + dw))  # 居中填充图片
        return image_paded


    # 变换函数
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 读取图片并预处理
    image = resize_padding(Image.open("./car.jpg"), 224)
    image = transform(image)
    image = image.reshape(1, 3, 224, 224)

    # 建立模型并恢复权重
    weight_path = "./checkpoint/convnext_tiny-983f1562.pth"
    pre_weights = torch.load(weight_path)
    model = convnext_tiny()
    model.load_state_dict(pre_weights)
    # print(model)

    # 单张图片推理
    model.cpu().eval()  # .eval()用于通知BN层和dropout层，采用推理模式而不是训练模式
    with torch.no_grad():  # torch.no_grad()用于整体修改模型中每一层的requires_grad属性，使得所有可训练参数不能修改，且正向计算时不保存中间过程，以节省内存
        output = torch.squeeze(model(image))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    # 输出结果
    print(predict_cla)
    print(predict[predict_cla])

# from functools import partial
# from typing import Any, Callable, List, Optional, Sequence
#
# import torch
# from torch import nn, Tensor
# from torch.nn import functional as F
#
# import sys
#
# sys.path.append('/mnt/llz/code/cls/mymodel/models/')
#
#
# from ops.misc import Conv2dNormActivation, Permute
# from ops.stochastic_depth import StochasticDepth
# from transforms._presets import ImageClassification
#
# from torchvision113.utils import _log_api_usage_once
# from torchvision113.models._api import WeightsEnum, Weights
# from torchvision113.models._meta import _IMAGENET_CATEGORIES
# from torchvision113.models._utils import handle_legacy_interface, _ovewrite_named_param
#
#
# __all__ = [
#     "ConvNeXt",
#     "ConvNeXt_Tiny_Weights",
#     "ConvNeXt_Small_Weights",
#     "ConvNeXt_Base_Weights",
#     "ConvNeXt_Large_Weights",
#     "convnext_tiny",
#     "convnext_small",
#     "convnext_base",
#     "convnext_large",
# ]
#
#
# class LayerNorm2d(nn.LayerNorm):
#     def forward(self, x: Tensor) -> Tensor:
#         x = x.permute(0, 2, 3, 1)
#         x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         x = x.permute(0, 3, 1, 2)
#         return x
#
#
# class CNBlock(nn.Module):
#     def __init__(
#         self,
#         dim,
#         layer_scale: float,
#         stochastic_depth_prob: float,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = partial(nn.LayerNorm, eps=1e-6)
#
#         self.block = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
#             Permute([0, 2, 3, 1]),
#             norm_layer(dim),
#             nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
#             nn.GELU(),
#             nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
#             Permute([0, 3, 1, 2]),
#         )
#         self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
#         self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
#
#     def forward(self, input: Tensor) -> Tensor:
#         result = self.layer_scale * self.block(input)
#         result = self.stochastic_depth(result)
#         result += input
#         return result
#
#
# class CNBlockConfig:
#     # Stores information listed at Section 3 of the ConvNeXt paper
#     def __init__(
#         self,
#         input_channels: int,
#         out_channels: Optional[int],
#         num_layers: int,
#     ) -> None:
#         self.input_channels = input_channels
#         self.out_channels = out_channels
#         self.num_layers = num_layers
#
#     def __repr__(self) -> str:
#         s = self.__class__.__name__ + "("
#         s += "input_channels={input_channels}"
#         s += ", out_channels={out_channels}"
#         s += ", num_layers={num_layers}"
#         s += ")"
#         return s.format(**self.__dict__)
#
#
# class ConvNeXt(nn.Module):
#     def __init__(
#         self,
#         block_setting: List[CNBlockConfig],
#         stochastic_depth_prob: float = 0.0,
#         layer_scale: float = 1e-6,
#         num_classes: int = 1000,
#         block: Optional[Callable[..., nn.Module]] = None,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#         **kwargs: Any,
#     ) -> None:
#         super().__init__()
#         _log_api_usage_once(self)
#
#         if not block_setting:
#             raise ValueError("The block_setting should not be empty")
#         elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
#             raise TypeError("The block_setting should be List[CNBlockConfig]")
#
#         if block is None:
#             block = CNBlock
#
#         if norm_layer is None:
#             norm_layer = partial(LayerNorm2d, eps=1e-6)
#
#         layers: List[nn.Module] = []
#
#         # Stem
#         firstconv_output_channels = block_setting[0].input_channels
#         layers.append(
#             Conv2dNormActivation(
#                 3,
#                 firstconv_output_channels,
#                 kernel_size=4,
#                 stride=4,
#                 padding=0,
#                 norm_layer=norm_layer,
#                 activation_layer=None,
#                 bias=True,
#             )
#         )
#
#         total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
#         stage_block_id = 0
#         for cnf in block_setting:
#             # Bottlenecks
#             stage: List[nn.Module] = []
#             for _ in range(cnf.num_layers):
#                 # adjust stochastic depth probability based on the depth of the stage block
#                 sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
#                 stage.append(block(cnf.input_channels, layer_scale, sd_prob))
#                 stage_block_id += 1
#             layers.append(nn.Sequential(*stage))
#             if cnf.out_channels is not None:
#                 # Downsampling
#                 layers.append(
#                     nn.Sequential(
#                         norm_layer(cnf.input_channels),
#                         nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
#                     )
#                 )
#
#         self.features = nn.Sequential(*layers)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#
#         lastblock = block_setting[-1]
#         lastconv_output_channels = (
#             lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
#         )
#         self.classifier = nn.Sequential(
#             norm_layer(lastconv_output_channels), nn.Flatten(1), nn.Linear(lastconv_output_channels, num_classes)
#         )
#
#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.Linear)):
#                 nn.init.trunc_normal_(m.weight, std=0.02)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#
#     def _forward_impl(self, x: Tensor) -> Tensor:
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = self.classifier(x)
#         return x
#
#     def forward(self, x: Tensor) -> Tensor:
#         return self._forward_impl(x)
#
#
# def _convnext(
#     block_setting: List[CNBlockConfig],
#     stochastic_depth_prob: float,
#     weights: Optional[WeightsEnum],
#     progress: bool,
#     **kwargs: Any,
# ) -> ConvNeXt:
#     if weights is not None:
#         _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
#
#     model = ConvNeXt(block_setting, stochastic_depth_prob=stochastic_depth_prob, **kwargs)
#
#     if weights is not None:
#         model.load_state_dict(weights.get_state_dict(progress=progress))
#
#     return model
#
#
# _COMMON_META = {
#     "min_size": (32, 32),
#     "categories": _IMAGENET_CATEGORIES,
#     "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#convnext",
#     "_docs": """
#         These weights improve upon the results of the original paper by using a modified version of TorchVision's
#         `new training recipe
#         <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
#     """,
# }
#
#
# class ConvNeXt_Tiny_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
#         transforms=partial(ImageClassification, crop_size=224, resize_size=236),
#         meta={
#             **_COMMON_META,
#             "num_params": 28589128,
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 82.520,
#                     "acc@5": 96.146,
#                 }
#             },
#         },
#     )
#     DEFAULT = IMAGENET1K_V1
#
#
# class ConvNeXt_Small_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/convnext_small-0c510722.pth",
#         transforms=partial(ImageClassification, crop_size=224, resize_size=230),
#         meta={
#             **_COMMON_META,
#             "num_params": 50223688,
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 83.616,
#                     "acc@5": 96.650,
#                 }
#             },
#         },
#     )
#     DEFAULT = IMAGENET1K_V1
#
#
# class ConvNeXt_Base_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/convnext_base-6075fbad.pth",
#         transforms=partial(ImageClassification, crop_size=224, resize_size=232),
#         meta={
#             **_COMMON_META,
#             "num_params": 88591464,
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 84.062,
#                     "acc@5": 96.870,
#                 }
#             },
#         },
#     )
#     DEFAULT = IMAGENET1K_V1
#
#
# class ConvNeXt_Large_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/convnext_large-ea097f82.pth",
#         transforms=partial(ImageClassification, crop_size=224, resize_size=232),
#         meta={
#             **_COMMON_META,
#             "num_params": 197767336,
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 84.414,
#                     "acc@5": 96.976,
#                 }
#             },
#         },
#     )
#     DEFAULT = IMAGENET1K_V1
#
#
# @handle_legacy_interface(weights=("pretrained", ConvNeXt_Tiny_Weights.IMAGENET1K_V1))
# def convnext_tiny(*, weights: Optional[ConvNeXt_Tiny_Weights] = None, progress: bool = True, **kwargs: Any) -> ConvNeXt:
#     """ConvNeXt Tiny model architecture from the
#     `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.
#
#     Args:
#         weights (:class:`~torchvision.models.convnext.ConvNeXt_Tiny_Weights`, optional): The pretrained
#             weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Tiny_Weights`
#             below for more details and possible values. By default, no pre-trained weights are used.
#         progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
#             for more details about this class.
#
#     .. autoclass:: torchvision.models.ConvNeXt_Tiny_Weights
#         :members:
#     """
#     weights = ConvNeXt_Tiny_Weights.verify(weights)
#
#     block_setting = [
#         CNBlockConfig(96, 192, 3),
#         CNBlockConfig(192, 384, 3),
#         CNBlockConfig(384, 768, 9),
#         CNBlockConfig(768, None, 3),
#     ]
#     stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
#     return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)
#
#
# @handle_legacy_interface(weights=("pretrained", ConvNeXt_Small_Weights.IMAGENET1K_V1))
# def convnext_small(
#     *, weights: Optional[ConvNeXt_Small_Weights] = None, progress: bool = True, **kwargs: Any
# ) -> ConvNeXt:
#     """ConvNeXt Small model architecture from the
#     `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.
#
#     Args:
#         weights (:class:`~torchvision.models.convnext.ConvNeXt_Small_Weights`, optional): The pretrained
#             weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Small_Weights`
#             below for more details and possible values. By default, no pre-trained weights are used.
#         progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
#             for more details about this class.
#
#     .. autoclass:: torchvision.models.ConvNeXt_Small_Weights
#         :members:
#     """
#     weights = ConvNeXt_Small_Weights.verify(weights)
#
#     block_setting = [
#         CNBlockConfig(96, 192, 3),
#         CNBlockConfig(192, 384, 3),
#         CNBlockConfig(384, 768, 27),
#         CNBlockConfig(768, None, 3),
#     ]
#     stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.4)
#     return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)
#
#
# @handle_legacy_interface(weights=("pretrained", ConvNeXt_Base_Weights.IMAGENET1K_V1))
# def convnext_base(*, weights: Optional[ConvNeXt_Base_Weights] = None, progress: bool = True, **kwargs: Any) -> ConvNeXt:
#     """ConvNeXt Base model architecture from the
#     `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.
#
#     Args:
#         weights (:class:`~torchvision.models.convnext.ConvNeXt_Base_Weights`, optional): The pretrained
#             weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Base_Weights`
#             below for more details and possible values. By default, no pre-trained weights are used.
#         progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
#             for more details about this class.
#
#     .. autoclass:: torchvision.models.ConvNeXt_Base_Weights
#         :members:
#     """
#     weights = ConvNeXt_Base_Weights.verify(weights)
#
#     block_setting = [
#         CNBlockConfig(128, 256, 3),
#         CNBlockConfig(256, 512, 3),
#         CNBlockConfig(512, 1024, 27),
#         CNBlockConfig(1024, None, 3),
#     ]
#     stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
#     return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)
#
#
# @handle_legacy_interface(weights=("pretrained", ConvNeXt_Large_Weights.IMAGENET1K_V1))
# def convnext_large(
#     *, weights: Optional[ConvNeXt_Large_Weights] = None, progress: bool = True, **kwargs: Any
# ) -> ConvNeXt:
#     """ConvNeXt Large model architecture from the
#     `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.
#
#     Args:
#         weights (:class:`~torchvision.models.convnext.ConvNeXt_Large_Weights`, optional): The pretrained
#             weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Large_Weights`
#             below for more details and possible values. By default, no pre-trained weights are used.
#         progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
#             for more details about this class.
#
#     .. autoclass:: torchvision.models.ConvNeXt_Large_Weights
#         :members:
#     """
#     weights = ConvNeXt_Large_Weights.verify(weights)
#
#     block_setting = [
#         CNBlockConfig(192, 384, 3),
#         CNBlockConfig(384, 768, 3),
#         CNBlockConfig(768, 1536, 27),
#         CNBlockConfig(1536, None, 3),
#     ]
#     stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
#     return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)

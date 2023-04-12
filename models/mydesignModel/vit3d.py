import torch
import torch.nn as nn
from einops import rearrange,reduce,repeat
from einops.layers.torch import Rearrange,Reduce
# from torchsummary import  summary
from torchinfo import summary
import os
from torch import Tensor
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 40, n_channels: int = 4, patch_size: int = 12, emb_size: int = 1024,
                 img_size=240):
        self.patch_size = patch_size
        self.n_channels = n_channels
        super().__init__()
        self.maxpool3d = nn.MaxPool3d(2)
        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b (h n_channels) (l s1) (w s2) -> b (h l w) (s1 s2 n_channels)', s1=patch_size, s2=patch_size,
                      n_channels=n_channels),
            # 注意这里的隐层大小设置的也是768，可以配置

            nn.Linear(patch_size * patch_size * n_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(
            torch.randn((((img_size // 2) // patch_size) ** 2) * ((in_channels // 2) // n_channels) + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.maxpool3d(x)

        x = self.projection(x)
        # print('after projection: x shape', x.shape)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        #         print(cls_tokens.shape)
        print(x.shape, self.positions.shape)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 1024, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        #         print("1qkv's shape: ", self.qkv(x).shape)
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        #         print("2qkv's shape: ", qkv.shape)

        queries, keys, values = qkv[0], qkv[1], qkv[2]
        #         print("queries's shape: ", queries.shape)
        #         print("keys's shape: ", keys.shape)
        #         print("values's shape: ", values.shape)

        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        #         print("energy's shape: ", energy.shape)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        #         print("scaling: ", scaling)
        att = F.softmax(energy, dim=-1) / scaling
        #         print("att1' shape: ", att.shape)
        att = self.att_drop(att)
        #         print("att2' shape: ", att.shape)

        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        #         print("out1's shape: ", out.shape)
        out = rearrange(out, "b h n d -> b n (h d)")
        #         print("out2's shape: ", out.shape)
        out = self.projection(out)
        #         print("out3's shape: ", out.shape)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 1024,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])



class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 1024, n_classes: int = 2):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))

class ViT(nn.Sequential):
    def __init__(self,
                in_channels: int = 40,
                 n_channels:int =4,
                patch_size: int = 12,
                emb_size: int = 1024,
                img_size: int = 240,
                depth: int = 12,
                n_classes: int = 2,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels,n_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )


if __name__ == '__main__':
    torch.cuda.empty_cache()
    device_ids = [0, 1]
    model = ViT()
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda(device=device_ids[1])
    summary(model, (1, 40, 240, 240))
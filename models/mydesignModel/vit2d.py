import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Compose,Resize,ToTensor
from einops import rearrange,reduce,repeat
from einops.layers.torch import Rearrange,Reduce
from torchsummary import summary
import os

class PatchEmbedding(nn.Module):
    def __init__(self,in_channels:int=3,patch_size:int=16,
                 emb_size:int=768,img_size:int=224):
        self.patch_size=patch_size
        super().__init__()
        self.projection=nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)',s1=patch_size,s2=patch_size),
            nn.Linear(patch_size*patch_size*in_channels,emb_size)
        )
        self.cls_token=nn.Parameter(torch.randn(1,1,emb_size))
        self.positions=nn.Parameter(torch.randn((img_size//patch_size)**2+1,emb_size))

    def forward(self,x):
        b,_,_,_=x.shape
        x=self.projection(x)
        cls_token=repeat(self.cls_token,'() n e -> b n e',b=b)
        x=torch.cat([cls_token,x],dim=1)
        x+=self.positions
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,emb_size=512,num_heads=8,dropout=0.):
        super().__init__()
        self.emb_size=emb_size
        self.num_heads=num_heads
        self.keys=nn.Linear(emb_size,emb_size)
        self.queries=nn.Linear(emb_size,emb_size)
        self.values=nn.Linear(emb_size,emb_size)
        self.att_drop=nn.Dropout(dropout)
        self.projection=nn.Linear(emb_size,emb_size)

    def forward(self,x,mask=None):
        queries=rearrange(self.queries(x),"b n (h d) -> b h n d",h=self.num_heads)
        keys=rearrange(self.keys(x),"b n (h d) -> b h n d",h=self.num_heads)
        values=rearrange(self.values(x),"b n (h d) -> b h n d",h=self.num_heads)

        energy=torch.einsum('bhqd, bhkd -> bhqk',queries,keys)
        if mask is not None:
            fill_value=torch.finfo(torch.float32).min
            energy.mask_fill(~mask,fill_value)

        scaling=self.emb_size**(1/2)
        att=F.softmax(energy,dim=-1)/scaling
        att=self.att_drop(att)
        print('att.shape is ',att.shape)
        print('values.shape is ', values.shape)
        out=torch.einsum('bhal,bhlv -> bhav',att,values)
        out=rearrange(out,'b h n d ->b n (h d)')
        out=self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn=fn

    def forward(self,x,**kwargs):
        residual=x
        x=self.fn(x,**kwargs)
        x+=residual
        return x

class FeedForwardBlock(nn.Module):
    def __init__(self,emb_size,expansion=4,drop_p=0.):
        super().__init__()
        self.ln=nn.Linear(emb_size,expansion*emb_size)
        self.gelu=nn.GELU()
        self.dropout=nn.Dropout(drop_p)
        self.ln2=nn.Linear(expansion*emb_size,emb_size)

    def forward(self,x):
        x=self.ln(x)
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.ln2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self,emb_size=768,drop_p=0.,forward_expansion=4,forward_drop_p=0.,**kwargs):
        super().__init__()
        self.residualBlock1=nn.Sequential(
            nn.LayerNorm(emb_size),
            MultiHeadAttention(emb_size, **kwargs),
            nn.Dropout(drop_p)
        )
        self.residualBlock2=nn.Sequential(
            nn.LayerNorm(emb_size),
            FeedForwardBlock(emb_size, expansion=forward_expansion,drop_p=forward_drop_p),
            nn.Dropout(drop_p)
        )

    def forward(self,x):
        x=self.residualBlock1(x)
        x=self.residualBlock2(x)
        return x

class ClassificationHead(nn.Module):
    def __init__(self,emb_size=768,n_classes=1000):
        super().__init__()
        self.reduce=Reduce('b n e -> b e',reduction='mean')
        self.layernorm=nn.LayerNorm(emb_size)
        self.ln=nn.Linear(emb_size,n_classes)

    def forward(self,x):
        x=self.reduce(x)
        x=self.layernorm(x)
        x=self.ln(x)
        return x


class TransformerEncoder(nn.Sequential):
    def __init__(self,depth=12,**kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])



class ViT(nn.Module):
    def __init__(self,
                 in_channels=3,
                 patch_size=16,
                 emb_size=768,
                 img_size=224,
                 depth=12,
                 n_classes=1000,**kwargs):
        super().__init__()
        self.patchembedding=PatchEmbedding(in_channels,patch_size,emb_size,img_size)
        self.encoder=TransformerEncoder(depth,emb_size=emb_size,**kwargs)
        self.cls=ClassificationHead(emb_size,n_classes)

    def forward(self,x):
        x=self.patchembedding(x)
        x=self.encoder(x)
        x=self.cls(x)
        return x



if __name__ == '__main__':
    vit = ViT().cuda()
    summary(vit, (3, 224, 224))




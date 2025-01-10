## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)



    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# Dual up-sample
class UpSample(nn.Module):
    def __init__(self,  in_channels):
        super(UpSample, self).__init__()

        self.up_p = nn.Sequential(nn.Conv2d(in_channels, 2*in_channels, 1, 1, 0, bias=False),
                                  nn.PReLU(),
                                  nn.PixelShuffle(2),
                                  nn.Conv2d(in_channels//2, in_channels//2, 1, stride=1, padding=0, bias=False))

        self.up_b = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                                  nn.PReLU(),
                                  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x_p = self.up_p(x)  # pixel shuffle
        x_b = self.up_b(x)  # bilinear
        out = torch.cat([x_p, x_b], dim=1)
        return out

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(192, LayerNorm_type)
        self.norm2 = LayerNorm(144, LayerNorm_type)
        self.norm3 = LayerNorm(84, LayerNorm_type)
        self.norm4 = LayerNorm(53, LayerNorm_type)
        self.attn1 = Attention(192, num_heads, bias)
        self.ffn1 = FeedForward(192, ffn_expansion_factor, bias)
        self.attn2 = Attention(144, num_heads, bias)
        self.ffn2 = FeedForward(144, ffn_expansion_factor, bias)
        self.attn3 = Attention(84, num_heads, bias)
        self.ffn3 = FeedForward(84, ffn_expansion_factor, bias)
        self.attn4 = Attention(53, num_heads, bias)
        self.ffn4 = FeedForward(53, ffn_expansion_factor, bias)
        self.upsample = nn.Sequential(nn.PixelShuffle(2),nn.PReLU())
        # self.upsample = UpSample(in_channels=dim, scale_factor=2)

    def forward(self, x,x1,x2,x3):

        # down  8x
        x3 = x3 + self.attn1(self.norm1(x3))
        x3 = x3 + self.ffn1(self.norm1(x3))
        x2 = torch.cat((self.upsample(x3),x2),1)

        # down  4x
        x2 = x2 + self.attn2(self.norm2(x2))
        x2 = x2 + self.ffn2(self.norm2(x2))
        x1 = torch.cat((self.upsample(x2),x1),1)

        # down  2x
        x1 = x1 + self.attn3(self.norm3(x1))
        x1 = x1 + self.ffn3(self.norm3(x1))
        x = torch.cat((self.upsample(x1),x),1)

        x = x + self.attn4(self.norm4(x))
        x = x + self.ffn4(self.norm4(x))


        return x

    # def forward(self, x):
    #     x = x + self.attn(self.norm1(x))
    #     x = x + self.ffn(self.norm2(x))
    #
    #     return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class RestormerQE(nn.Module):
    def __init__(self,
                 dim=32,
                 heads=[1],
                 ffn_expansion_factor=1,
                 bias=False,
                 LayerNorm_type='BiasFree',  ## Other option 'BiasFree'
                 ):

        super(RestormerQE, self).__init__()

        self.encoder_level2 =TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type)
        self.out = nn.Conv2d(53, 1, kernel_size=3, padding=1)

    def forward(self, x,x1,x2,x3):
        out = self.encoder_level2(x,x1,x2,x3)
        out = self.out(out)
        return out

if __name__ == '__main__':
    height = 1280
    width = 720
    x = torch.randn((1, 32, height, width)).cuda()
    x1 = torch.randn((1, 48, height//2, width//2)).cuda()
    x2 = torch.randn((1, 96, height//4, width//4)).cuda()
    x3 = torch.randn((1, 192, height//8, width//8)).cuda()
    model = RestormerQE(dim=32).cuda()
    print(model)
    print('{:>16s} : {:<.4f} [M]'.format('#Params', sum(map(lambda x: x.numel(), model.parameters())) / 10 ** 6))
    print('input image size: (%d, %d)' % (height, width))
    # print('FLOPs: %.4f G' % (model.flops() / 1e9))
    # print('model parameters: ', network_parameters(model))
    time_start = time.time()
    with torch.no_grad():
        model = model.eval()
        x = model(x,x1,x2,x3)
    print('output image size: ', x.shape)
    # flops, params = profile(model, (x,))
    # print(flops)
    # print(params)
    time_end = time.time()
    time_c = time_end - time_start
    print(x.shape,'time cost:{},s'.format('%.3f' %time_c))

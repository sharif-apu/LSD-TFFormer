import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
# import cv2


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')
class IG_MSA_M(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=4,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        #print(x_in.shape, illu_fea_trans.shape, "attention")
        b, hw, c = x_in.shape
        x = x_in#.reshape(b, h * w, c)
        #print("query shape",x.shape )
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans # illu_fea: b,c,h,w -> b,h,w,c
        #print("query shape",x.shape, illu_attn.shape )

        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn))
        
        #print("v shape", q.shape, k.shape, v.shape, illu_attn.shape, self.num_heads)
        v = v * illu_attn
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, hw, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, hw, c)
        #print("ppppp", out_c.shape, v_inp.reshape(b, 1 , hw, c).permute(0, 3, 2, 1).shape)
        out_p = self.pos_emb(v_inp.reshape(b, 1, hw, c).permute(0, 3, 2, 1)).reshape(b,hw,c)#.permute(0, 1, 2)
        #print("ppppp", out_p.shape)
        out = out_c + out_p
        #print(out.shape)
        return out
class AOA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=80,
            heads=4,
            num_blocks=1,
    ):
        super().__init__()
        self.chrom = IG_MSA(dim, dim_head, heads)
        self.c_norm =  nn.LayerNorm(dim)#PreNorm(dim, FeedForward(dim=dim))
        self.lum = IG_MSA(dim, dim_head, heads)
        self.l_norm =  nn.LayerNorm(dim)#PreNorm(dim, FeedForward(dim=dim))
        self.cross = IG_MSA(dim, dim_head, heads)
        self.cr_norm =  nn.LayerNorm(dim)#PreNorm(dim, FeedForward(dim=dim))

    def forward(self, lum, chrom,  illu_fea, chrom_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        #print(chrom.shape)
        c = self.c_norm(self.chrom(chrom.permute(0, 2, 3, 1), chrom_fea.permute(0, 2, 3, 1)))
        l = self.l_norm(self.lum(lum.permute(0, 2, 3, 1), illu_fea.permute(0, 2, 3, 1)))
        out = self.cr_norm(self.cross(c,l))
        #print(lum.shape, out.shape)
        return out.permute(0,3, 1, 2)
    

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# input [bs,28,256,310]  output [bs, 28, 256, 256]
def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]


class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img, ill):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        
        mean_c = ill#img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img,mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map

class chrom_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(chrom_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(6, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img, chrom):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        
        #mean_c = chrom.mean(dim=1).unsqueeze(1)
        # stx()
        #print(chrom.shape, img.shape)
        input = torch.cat([img,chrom], dim=1)
        #print(input.shape)
        x_1 = self.conv1(input)
        #print(x_1.shape)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map



class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        #print(x_in.shape, illu_fea_trans.shape, "attention")
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        #print("query shape",x.shape )
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans # illu_fea: b,c,h,w -> b,h,w,c
        #print("query shape",x.shape, illu_attn.shape )

        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        
        #print("v shape", q.shape, k.shape, v.shape, illu_attn.shape, self.num_heads)
        v = v * illu_attn
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        #print("c shape", out_c.shape, v_inp.reshape(b, h, w, c).permute(
        #    0, 3, 1, 2).shape)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        #print( "pos_emb",out_p.shape, v_inp.reshape(b, h, w, c).permute(
        #    0, 3, 1, 2).shape)
        out = out_c + out_p
        #print(out.shape)
        return out



class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        #print("IGAB", self.blocks)
        #print(x.shape, illu_fea.shape)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        #print()
        return out


class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)
     
        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            print("IGAB", dim_level, dim)
            self.encoder_layers.append(nn.ModuleList([
                IGAB(
                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(
            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                IGAB(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)
         
        
        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea,illu_fea)  # bchw
            print("deature and input shape",fea.shape, illu_fea.shape)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea,illu_fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level-1-i]
            fea = LeWinBlcok(fea,illu_fea)

        # Mapping
        out = self.mapping(fea) + x

        return out

def lum_chrom(x):
    red_channel = x[:, 0:1, :, :]   # Extracting the first channel (Red)
    green_channel = x[:, 1:2, :, :] # Extracting the second channel (Green)
    blue_channel = x[:, 2:3, :, :]  # Extracting the th
    luminance = 0.299 * red_channel + 0.587 * green_channel + 0.114 * blue_channel
    chrome =  x[:, 0:3, :, :] - luminance
    return luminance, chrome

class RetinexFormer_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, level=2, num_blocks=[1, 1, 1]):
        super(RetinexFormer_Single_Stage, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels,out_dim=out_channels,dim=n_feat,level=level,num_blocks=num_blocks)  #### 将 Denoiser 改为 img2img
    
    def forward(self, img):
        # img:        b,c=3,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        output_img = self.denoiser(input_img,illu_fea)

        return output_img

def get_attention_config(dim, heads=4):
    dim_head = dim // heads
    return dim_head, heads

class TwoTineFormerN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, num_blocks=[1, 1, 1]):
        super(TwoTineFormerN, self).__init__()
        dim = n_feat

        self.lum_estimator = Illumination_Estimator(n_feat)
        self.chrom_estimator = chrom_Estimator(n_feat)

        self.lum_embedding = nn.Conv2d(3, dim, 3, 1, 1, bias=False)
        self.chrom_embedding = nn.Conv2d(3, dim, 3, 1, 1, bias=False)

        dim_head, heads = get_attention_config(dim)
        self.encoder1 = IGAB(dim=dim, dim_head=dim_head, heads=heads, num_blocks=num_blocks[0])
        self.down1 = nn.Conv2d(dim, dim * 2, 4, 2, 1, bias=False)
        self.fea_down1 = nn.Conv2d(dim, dim * 2, 4, 2, 1, bias=False)

        dim2 = dim * 2
        dim_head2, heads2 = get_attention_config(dim2)
        self.encoder2 = IGAB(dim=dim2, dim_head=dim_head2, heads=heads2, num_blocks=num_blocks[1])
        self.down2 = nn.Conv2d(dim2, dim2 * 2, 4, 2, 1, bias=False)
        self.fea_down2 = nn.Conv2d(dim2, dim2 * 2, 4, 2, 1, bias=False)

        dim4 = dim2 * 2
        dim_head4, heads4 = get_attention_config(dim4)
        self.encoder3 = IGAB(dim=dim4, dim_head=dim_head4, heads=heads4, num_blocks=num_blocks[1])
        self.down3 = nn.Conv2d(dim4, dim4 * 2, 4, 2, 1, bias=False)
        self.fea_down3 = nn.Conv2d(dim4, dim4 * 2, 4, 2, 1, bias=False)

        dim8 = dim4 * 2
        dim_head8, heads8 = get_attention_config(dim8)
        self.aoa = AOA(dim8)
        self.middle = IGAB(dim=dim8, dim_head=dim_head8, heads=heads8, num_blocks=num_blocks[1])

        self.dup1 = nn.ConvTranspose2d(dim8, dim4, kernel_size=2, stride=2)
        self.decoder1 = IGAB(dim=dim4, dim_head=dim_head4, heads=heads4, num_blocks=num_blocks[1])

        self.dup2 = nn.ConvTranspose2d(dim4, dim2, kernel_size=2, stride=2)
        self.decoder2 = IGAB(dim=dim2, dim_head=dim_head2, heads=heads2, num_blocks=num_blocks[1])

        self.dup3 = nn.ConvTranspose2d(dim2, dim, kernel_size=2, stride=2)
        self.decoder3 = IGAB(dim=dim, dim_head=dim_head, heads=heads, num_blocks=num_blocks[1])

        self.recon = nn.Conv2d(dim, 3, 3, 1, 1, bias=False)

        self.lum_embedding_r = nn.Conv2d(1, dim // 2, 3, 1, 1, bias=False)
        self.chrom_embedding_r = nn.Conv2d(3, dim // 2, 3, 1, 1, bias=False)
        self.r1 = IGAB(dim=dim, dim_head=dim_head, heads=heads, num_blocks=4)

        self.out = nn.Conv2d(dim, 3, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, img):
        l, c = lum_chrom(img)

        illu_fea, illu_map = self.lum_estimator(img, l)
        chrom_fea, chrom_map = self.chrom_estimator(img, c)

        input_img_lum = img * illu_map + img
        input_img_chrom = img * chrom_map + img

        lum_emb = self.lrelu(self.lum_embedding(input_img_lum))
        chrom_emb = self.lrelu(self.chrom_embedding(input_img_chrom))

        lum_enc1 = self.encoder1(lum_emb, illu_fea)
        lum_down1 = self.lrelu(self.down1(lum_enc1))
        lum_fea_down1 = self.lrelu(self.fea_down1(illu_fea))

        chrom_enc1 = self.encoder1(chrom_emb, chrom_fea)
        chrom_down1 = self.lrelu(self.down1(chrom_enc1))
        chrom_fea_down1 = self.lrelu(self.fea_down1(chrom_fea))

        lum_enc2 = self.encoder2(lum_down1, lum_fea_down1)
        lum_down2 = self.lrelu(self.down2(lum_enc2))
        lum_fea_down2 = self.lrelu(self.fea_down2(lum_fea_down1))

        chrom_enc2 = self.encoder2(chrom_down1, chrom_fea_down1)
        chrom_down2 = self.lrelu(self.down2(chrom_enc2))
        chrom_fea_down2 = self.lrelu(self.fea_down2(chrom_fea_down1))

        lum_enc3 = self.encoder3(lum_down2, lum_fea_down2)
        lum_down3 = self.lrelu(self.down3(lum_enc3))
        lum_fea_down3 = self.lrelu(self.fea_down3(lum_fea_down2))

        chrom_enc3 = self.encoder3(chrom_down2, chrom_fea_down2)
        chrom_down3 = self.lrelu(self.down3(chrom_enc3))
        chrom_fea_down3 = self.lrelu(self.fea_down3(chrom_fea_down2))

        aoa = self.aoa(lum_down3, chrom_down3, lum_fea_down3, chrom_fea_down3)
        #print(lum_down3.shape)
        mid = self.middle( aoa, lum_down3 + chrom_down3)

        up1 = self.lrelu(self.dup1(mid))
        dec1 = self.decoder1(up1, chrom_fea_down2 + lum_fea_down2)

        up2 = self.lrelu(self.dup2(dec1))
        dec2 = self.decoder2(up2, chrom_fea_down1 + lum_fea_down1)

        up3 = self.lrelu(self.dup3(dec2))
        #print(lum_down1.shape)
        dec3 = self.decoder3(up3, chrom_fea + illu_fea)

        recon = self.recon(dec3) + img

        l_r, c_r = lum_chrom(recon)
        l_r_emb = self.lrelu(self.lum_embedding_r(l_r))
        c_r_emb = self.lrelu(self.chrom_embedding_r(c_r))
        emb = torch.cat((l_r_emb, c_r_emb), 1)

        r1 = self.r1(dec3, emb)

        out = torch.tanh(self.out(r1) + recon)
        return recon, out


import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
import matplotlib.cm as cm
def hook_fn(name):
    def hook(module, input, output):
        features[name] = output
    return hook

cmapT = 'hsv'

def visualize_heatmap(tensor, title="Feature Map"):
    fmap = tensor[0].mean(0).detach().cpu()  # [H, W]
    plt.imshow(fmap, cmap=cmapT)
    # plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_channel_grid_color(tensor, title="Feature Channels", num_cols=8, cmap=cmapT):
    tensor = tensor[0]  # [C, H, W]
    
    # Normalize each channel individually
    tensor = (tensor - tensor.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]) / \
             (tensor.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0] + 1e-5)
    
    # Convert each 2D channel to a 3-channel RGB image using a colormap
    color_channels = []
    for c in tensor:
        c_np = c.cpu().numpy()
        c_color = cm.get_cmap(cmap)(c_np)[:, :, :3]  # Get RGB only (ignore alpha)
        c_color = torch.tensor(c_color).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        color_channels.append(c_color)

    # Stack and grid them
    color_tensor = torch.stack(color_channels)
    grid = make_grid(color_tensor, nrow=num_cols, pad_value=1)

    img = TF.to_pil_image(grid)
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_channel_grid(tensor, title="Feature Channels", num_cols=8):
    tensor = tensor[0]  # [C, H, W]
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-5)
    grid = make_grid(tensor.unsqueeze(1), nrow=num_cols, normalize=True, pad_value=1)
    img = TF.to_pil_image(grid)
    plt.figure(figsize=(12, 12))
    plt.imshow(img,cmap='viridis')
    plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    from torchsummary import summary
    model = RetinexFormer_Single_Stage().cuda()
    indim = 1024
    #print(model)
    inputs = torch.randn((1, 3, indim, indim )).cuda()
    import time
    start_time = time.time()
    for i in range(0,100):
        with torch.no_grad():
            out = model(inputs)
    endtime = ((time.time() - start_time)/100) * 1000
    #inferencetime = endtime
        
    with torch.no_grad():
        out = model(inputs)
        flops = FlopCountAnalysis(model,inputs)
        n_param = sum([p.nelement() for p in model.parameters()])  # 所有参数数量
        # print(f'GMac:{flops.total()/(1024*1024*1024)}')
        # print(f'Params:{n_param}')
        # #summary(model, input_size = (3, 128, 128))
        
        from ptflops import get_model_complexity_info
        from thop import profile
        macs, params = get_model_complexity_info(model, (3, indim, indim), as_strings=True,
                                                print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        print("inferen time (ms)", endtime)

        # dummy_input = torch.randn(1, 3, 256, 256).cuda()  # Adjust the size according to your model input

        # # Use thop to profile the model and calculate FLOPs
        # flops, params = profile(model, inputs=(dummy_input,))

        # print(f"Number of FLOPs: {flops}")
        # print(f"Number of parameters: {params}")

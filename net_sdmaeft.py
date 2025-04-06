import einops
from torch.nn import (
    Conv2d,
    BatchNorm2d,
    PReLU,
    Module
)
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from metrics import ArcFace, CosFace, AirFace, SphereFace, ArcNegFace, QAMFace, CircleLoss
from timm.models.layers import DropPath
from functools import partial
from layers import get_norm_act_layer, to_2tuple
from utils import _init_conv,named_apply

class GDC(nn.Module):
    def __init__(self, in_c, embedding_size):
        super(GDC, self).__init__()
        self.conv_6_dw = Linear_block(
            in_c, in_c, groups=in_c, kernel=(8, 32), stride=(1, 1), padding=(0, 0)
        )
        self.conv_6_flatten = Flatten()
        self.linear = nn.Linear(in_c, embedding_size, bias=False)
        # self.bn = BatchNorm1d(embedding_size, affine=False)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x
    
def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def num_groups(group_size, channels):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size

class CDilated(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1, bias=False):
        super().__init__()
        padding1 = int((kSize[0] - 1) / 2) * d
        padding2 = int((kSize[1] - 1) / 2) * d

        self.padding = nn.ConstantPad2d((padding2, padding2, padding1, padding1), 0.0)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, bias=bias, dilation=d, groups=nIn)

    def forward(self, input):

        output = self.padding(input)
        output = self.conv(output)
        return output


class TFSA(nn.Module):
    def __init__(self, c=64, causal=True):
        super(TFSA, self).__init__()
        self.d_c = c//4
        self.f_qkv = nn.Sequential(
            nn.Conv2d(c, self.d_c*3, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(self.d_c*3),
            nn.PReLU(self.d_c*3),
        )
        self.t_qk = nn.Sequential(
            nn.Conv2d(c, self.d_c*2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(self.d_c*2),
            nn.PReLU(self.d_c*2),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(self.d_c, c, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(c),
            nn.PReLU(c),
        )
        self.causal = causal

    def forward(self, inp):
        """
        inp: B C F T
        """
        # f-attention
        f_qkv = self.f_qkv(inp)
        qf, kf, v = tuple(einops.rearrange(
            f_qkv, "b (c k) f t->k b c f t", k=3))
        f_score = torch.einsum("bcft,bcyt->btfy", qf, kf) / (self.d_c**0.5)
        f_score = f_score.softmax(dim=-1)
        f_out = torch.einsum('btfy,bcyt->bcft', [f_score, v])
        # t-attention
        t_qk = self.t_qk(inp)
        qt, kt = tuple(einops.rearrange(t_qk, "b (c k) f t->k b c f t", k=2))
        t_score = torch.einsum('bcft,bcfy->bfty', [qt, kt]) / (self.d_c**0.5)
        mask_value = max_neg_value(t_score)
        if self.causal:
            i, j = t_score.shape[-2:]
            mask = torch.ones(i, j, device=t_score.device).triu_(j - i + 1).bool()
            t_score.masked_fill_(mask, mask_value)
        t_score = t_score.softmax(dim=-1)
        t_out = torch.einsum('bfty,bcfy->bcft', [t_score, f_out])
        out = self.proj(t_out)
        return out + inp


class TFDDC(nn.Module):

    def __init__(
            self,
            in_chs: int,
            expan_ratio:int = 4,
            stride: int = 1,
            dila:int = 1,
            drop_path: float = 0.,
    ):
        super(TFDDC, self).__init__()

        if stride == 2:
            self.skip_path = nn.Sequential(
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
                nn.Conv2d(in_channels=in_chs, out_channels=in_chs, kernel_size=(1, 1))
            )
        else:
            self.skip_path = nn.Identity()

        expan_ichs = in_chs * expan_ratio

        self.pconv1 = nn.Conv2d(in_chs, expan_ichs, kernel_size=(1, 1))
        self.norm_act1 = nn.Sequential(nn.BatchNorm2d(expan_ichs), nn.GELU())

        self.conv2_kxk_t = CDilated(expan_ichs, expan_ichs, kSize=(3, 5), stride=stride, d=dila, groups=expan_ichs)
        self.conv2_kxk_f = CDilated(expan_ichs, expan_ichs, kSize=(5, 3), stride=stride, d=dila, groups=expan_ichs)
        self.t_f_weight = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.norm_act2 = nn.Sequential(nn.BatchNorm2d(in_chs), nn.GELU())

        self.pconv2 = nn.Conv2d(expan_ichs, in_chs, kernel_size=(1, 1))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def init_weights(self, scheme=''):
        named_apply(partial(_init_conv, scheme=scheme), self)

    def forward(self, x):
        skip_path = self.skip_path(x)

        # pw1
        x = self.pconv1(x)
        x = self.norm_act1(x)

        # tfdw
        x_t = self.conv2_kxk_t(x)
        x_f = self.conv2_kxk_f(x)
        x = (self.t_f_weight * x_t) + ((1 - self.t_f_weight) * x_f)

        # pw2
        x = self.pconv2(x)
        # x = self.norm_act2(x)
        x = self.drop_path(x) + skip_path
        return x


class TFFI(nn.Module):
    """
    Local-Global Features Interaction
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6):
        super().__init__()

        self.dim = dim

        self.tfsa = TFSA(self.dim)
        self.norm = LayerNorm(self.dim, eps=1e-6)
        self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_ = x

        B, C, H, W = x.shape
        # ca
        x = self.tfsa(x)
        x = x.reshape(B, H, W, C)

        # Inverted Bottleneck
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input_ + self.drop_path(x)

        return x



class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        # print("kernel, ",self.weight.shape)
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Stem(nn.Module):

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int = 3,
            padding: str = '',
            bias: bool = False,
            act_layer: str = 'gelu',
            norm_layer: str = 'batchnorm2d',
            norm_eps: float = 1e-5,
            t_f_kernel=None,
    ):
        super().__init__()
        if not isinstance(out_chs, (list, tuple)):
            out_chs = to_2tuple(out_chs)

        norm_act_layer = partial(get_norm_act_layer(norm_layer, act_layer), eps=norm_eps)
        self.out_chs = out_chs[-1]
        self.stride = 2

        self.t_f_weight = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)

        t_kernel = t_f_kernel[0]
        f_kernel = t_f_kernel[1]

        self.conv1_t = Conv2dSame(in_channels=in_chs, out_channels=out_chs[0], kernel_size=t_kernel, stride=(2, 2), groups=1, bias=bias)
        self.conv1_f = Conv2dSame(in_channels=in_chs, out_channels=out_chs[0], kernel_size=f_kernel, stride=(2, 2), groups=1, bias=bias)

        self.norm1_t = norm_act_layer(out_chs[0])
        self.norm1_f = norm_act_layer(out_chs[0])

        self.conv2_t = torch.nn.Conv2d(in_channels=out_chs[0], out_channels=out_chs[0], kernel_size=t_kernel, padding=(2, 1), stride=1, groups=1, bias=bias)
        self.conv2_f = torch.nn.Conv2d(in_channels=out_chs[0], out_channels=out_chs[0], kernel_size=f_kernel, padding=(1, 2), stride=1, groups=1, bias=bias)

    def init_weights(self, scheme=''):
        named_apply(partial(_init_conv, scheme=scheme), self)

    def forward(self, x):

        x_t = self.conv2_t(self.norm1_t(self.conv1_t(x)))
        x_f = self.conv2_f(self.norm1_f(self.conv1_f(x)))

        return (x_t*self.t_f_weight) + (x_f*(1-self.t_f_weight))


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class BNGELU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-5)
        self.act = nn.GELU()

    def forward(self, x):
        output = self.bn(x)
        output = self.act(output)

        return output


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding=0, dilation=(1, 1), groups=1, bn_act=False, bias=False):
        super().__init__()

        self.bn_act = bn_act

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_act:
            self.bn_gelu = BNGELU(nOut)

    def forward(self, x):
        output = self.conv(x)

        if self.bn_act:
            output = self.bn_gelu(output)

        return output



class TgramNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=313):
        super(TgramNet, self).__init__()
        # if "center=True" of stft, padding = win_len / 2
        self.conv_extrctor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                # 313(10) , 63(2), 126(4)
                nn.LayerNorm(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False),
            ) for idx in range(num_layer)])

    def forward(self, x):
        out = self.conv_extrctor(x)
        out = self.conv_encoder(out)
        return out


class SDMAE_FT(nn.Module):
    def __init__(self, num_classes,
                 c_dim=128,
                 win_len=1024,
                 hop_len=313,
                 head='arcface'):
        super(SDMAE_FT, self).__init__()

        if head == 'arcface':
            self.head = ArcFace(in_features=128, out_features=num_classes, s=30, m_arc=0.7)
        elif head == 'cosface':
            self.head = CosFace(in_features=128, out_features=num_classes, s=30, m=0.7)
        elif head == 'airface':
            self.head = AirFace(in_features=128, out_features=num_classes, s=30, m=0.7)
        elif head == 'sphereface':
            self.head = SphereFace(in_features=128, out_features=num_classes, m=4.0)
        if head == 'arcnegface':
            self.head = ArcNegFace(in_features=128, out_features=num_classes)
        elif head == 'qamface':
            self.head = QAMFace(in_features=128, out_features=num_classes, s=30, m=0.5)
        elif head == 'circleloss':
            self.head = CircleLoss(in_features=128, out_features=num_classes)

        self.tgramnet = TgramNet(mel_bins=c_dim, win_len=win_len, hop_len=hop_len)

        self.ASD_encoder = ASD_Encoder()

    def get_tgram(self, x_wav):
        return self.tgramnet(x_wav)

    def forward(self, x_wav, x_mel, label=None):
        x_wav, x_mel = x_wav.unsqueeze(1), x_mel.unsqueeze(1)
        x_t = self.tgramnet(x_wav).unsqueeze(1)
        x = torch.cat((x_mel, x_t), dim=1)
        # x = x.unsqueeze(1)
        feature = self.ASD_encoder(x)
        if self.head:
            out = self.head(feature, label)
        return out, feature



class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


##################################  MobileFaceNet #############################################################


class Conv_block(Module):
    def __init__(
        self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1
    ):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(
            in_c,
            out_channels=out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(Module):
    def __init__(
        self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1
    ):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(
            in_c,
            out_channels=out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x



class ASD_Encoder(Module):
    def __init__(self,
                 global_block=[1, 1, 1],
                 drop_path_rate=0.,
                 global_block_type=['TFFI', 'TFFI', 'TFFI'],
                 expan_ratio=4,
    ):
        super(ASD_Encoder, self).__init__()

        self.depth = [4, 4, 4]
        self.dims = [128, 200, 320]
        self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]

        self.stem = Stem(in_chs=2, out_chs=128, padding='same',bias=True, norm_layer='batchnorm2d',
                         norm_eps=0.001, t_f_kernel=[(5, 3), (3, 5)])


        self.downsample_layers = nn.ModuleList()

        stem2 = nn.Sequential(
            Conv(self.dims[0], self.dims[0], kSize=3, stride=2, padding=1, bn_act=False),
        )
        self.downsample_layers.append(stem2)

        for i in range(2):
            downsample_layer = nn.Sequential(
                Conv(self.dims[i], self.dims[i+1], kSize=3, stride=2, padding=1, bn_act=False),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depth))]
        cur = 0

        for i in range(3):
            stage_blocks = []
            for j in range(self.depth[i]):
                if j > self.depth[i] - global_block[i] - 1:
                    if global_block_type[i] == 'TFFI':
                        stage_blocks.append(TFFI(dim=self.dims[i], drop_path=dp_rates[cur + j], expan_ratio=expan_ratio))

                    else:
                        raise NotImplementedError
                else:
                    stage_blocks.append(TFDDC(in_chs=self.dims[i],expan_ratio=expan_ratio,
                                                       stride=1, dila=self.dilation[i][j], drop_path=dp_rates[cur + j]))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.depth[i]

        self.conv1 = Conv_block(320, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))

        self.output_layer = GDC(512, 128)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)

        x = self.downsample_layers[0](x)
        for s in range(len(self.stages[0])-1):
            x = self.stages[0][s](x)
        x = self.stages[0][-1](x)


        for i in range(1, 3):
            x = self.downsample_layers[i](x)
            for s in range(len(self.stages[i]) - 1):
                x = self.stages[i][s](x)
            x = self.stages[i][-1](x)

        x = self.conv1(x)
        x = self.output_layer(x)

        return x





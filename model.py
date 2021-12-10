import torch
import torch.nn as nn
import torch.nn.functional as F
import PixelUnShuffle
from pytorch_wavelets import DWTForward, DWTInverse
import numpy as np
import math
############################################################################################
# Base models
############################################################################################
class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DeconvBlock, self).__init__()

        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.deconv(x)

        return self.act(out)
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class UpBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(UpBlock, self).__init__()

        self.conv1 = DeconvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = ConvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        hr = self.conv1(x)
        lr = self.conv2(hr)
        residue = self.local_weight1(x) - lr
        h_residue = self.conv3(residue)
        hr_weight = self.local_weight2(hr)
        return hr_weight + h_residue

class DownBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=1, stride=1, padding=1):
        super(DownBlock, self).__init__()
        self.conv1 = ConvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = ConvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x):
        lr = self.conv1(x)
        hr = self.conv2(lr)
        residue = self.local_weight1(x) - hr
        l_residue = self.conv3(residue)
        lr_weight = self.local_weight2(lr)
        return lr_weight + l_residue

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True, isuseBN=False,isusePL=True):
        super(ConvBlock, self).__init__()
        self.isuseBN = isuseBN
        self.isusePL = isusePL
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        if self.isuseBN:
            self.bn = nn.BatchNorm2d(output_size)
        if self.isusePL:
            self.act = torch.nn.PReLU()
    def forward(self, x):
        out = self.conv(x)
        if self.isuseBN:
            out = self.bn(out)
        if self.isusePL:
            out = self.act(out)
        return out

class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(num_filter)

        self.act1 = torch.nn.PReLU()
        self.act2 = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = self.act2(out)

        return out

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
class spatial_attn_layer(nn.Module):
        def __init__(self, kernel_size=5):
            super(spatial_attn_layer, self).__init__()
            self.compress = ChannelPool()
            self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

        def forward(self, x):
            # import pdb;pdb.set_trace()
            x_compress = self.compress(x)
            x_out = self.spatial(x_compress)
            scale = torch.sigmoid(x_out)  # broadcasting
            return x * scale
class GSA(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=1, stride=1, padding=0):
        super(GSA, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvBlock(input_size, output_size//2, kernel_size, stride, padding, bias=True,isusePL=False)
        self.conv2 = ConvBlock(input_size+output_size//2, output_size, kernel_size, stride, padding, bias=True,isusePL=False)
        self.resize = nn.functional.interpolate
        self.spa=spatial_attn_layer()
    def forward(self, x):
        p1=self.max_pool(x)
        p2=self.avg_pool(x)
        p1=(p1+p2)/2
        p1=self.resize(p1,size=[x.size()[2],x.size()[3]],scale_factor=None, mode='nearest')
        p2=self.conv1(p1)
        p5=self.spa(p2)
        p3=torch.cat((x, p5), 1)
        p4=self.conv2(p3)
        return p4
class ca_layer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(ca_layer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out
class MGDB(torch.nn.Module):
    def __init__(self, input_size,num_filter, kernel_size=3, stride=1, padding=1, bias=True):
        super(MGDB, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_size, num_filter, kernel_size, stride, 4*padding, bias=bias,dilation=4)
        # self.bn1 = nn.BatchNorm2d(num_filter)
        self.conv2 = torch.nn.Conv2d(input_size, num_filter,kernel_size, stride, 2*padding, bias=bias,dilation=2)
        # self.bn2 = nn.BatchNorm2d(num_filter)
        self.conv3 = torch.nn.Conv2d(input_size, num_filter, kernel_size, stride, padding, bias=bias,dilation=1)
        # self.bn3 = nn.BatchNorm2d(num_filter)
        self.conv4 = torch.nn.Conv2d(2*num_filter,2*num_filter, kernel_size, stride, 2*padding, bias=bias,dilation=2)
        # self.bn4 = nn.BatchNorm2d(num_filter)
        self.conv5 = torch.nn.Conv2d(input_size+num_filter, num_filter, kernel_size, stride, padding, bias=bias,dilation=1)
        self.conv6 = torch.nn.Conv2d(3*num_filter, num_filter, kernel_size, stride, padding, bias=bias,dilation=1)
        self.act1 = torch.nn.PReLU()
        self.act2 = torch.nn.PReLU()
        self.act3 = torch.nn.PReLU()
        self.act4 = torch.nn.PReLU()
        self.act5 = torch.nn.PReLU()
        self.act6 = torch.nn.PReLU()
        # self.CA=sa_layer(num_filter)
        self.CA=ca_layer(num_filter)
    def forward(self, x):
        out = self.conv1(x)
        out= self.act1(out)
        # out = self.bn1(out)
        out1 = self.conv2(x)
        out1 = self.act2(out1)
        # out1 = self.bn2(out1)
        out2=torch.cat((out,out1),axis=1)
        out3 = self.conv3(x)
        out3 = self.act3(out3)
        out3=torch.cat((x,out3),axis=1)
        out5 = self.conv5(out3)
        out5 = self.act5(out5)
        # out3 = self.bn3(out3)
        out6=self.conv4(out2)
        out6=self.act4(out6)
        out7=torch.cat((out5,out6),1)
        # out5 = self.conv4(out4)
        out8 = self.conv6(out7)
        out8=self.act6(out8)
        # out5 = self.bn4(out5)
        out8=self.CA(out8)
        return out8
class BSWN(nn.Module):
    def __init__(self, input_dim=3, dim=32):
        super(BSWN,self).__init__()
        self.DWT = DWTForward(J=1, wave='haar').cuda()  # J是分解的层数，wave为选择的小波类型
        self.IDWT = DWTInverse(wave='haar').cuda()
        self.feat1 = ConvBlock(input_dim, dim,3, 1, 1)
        self.feat2 = ConvBlock(4*dim, 2 * dim, 3, 1, 1)#128-64
        self.feat3 = ConvBlock(8*dim, 4 * dim, 3, 1, 1)
        self.feat4 = ConvBlock(16 * dim, 8 * dim, 3, 1, 1)
        self.GSA=GSA(input_size=8 * dim,output_size=8 * dim)
        self.feat5 = ConvBlock(2* dim, 4 * dim, 1, 1, 0)
        self.MGDB_1 = MGDB(input_size=4 * dim,num_filter=4 * dim)
        self.feat6 = ConvBlock(8 * dim, 4 * dim, 3, 1, 1)
        self.feat7 = ConvBlock(dim, 2 * dim, 1, 1, 0)
        self.MGDB_2= MGDB(input_size=2 * dim,num_filter=2 * dim)
        self.feat8 = ConvBlock(4 * dim, 2 * dim, 3, 1, 1)
        self.feat9 = ConvBlock(dim//2,  dim, 1, 1, 0)
        self.MGDB_3= MGDB(input_size=2*dim,num_filter=4 * dim)
        self.MGDB_4 = MGDB(input_size=6*dim,num_filter=4 * dim)
        self.feat10 = ConvBlock(10*dim,4*dim, 3, 1, 1)
        self.feat11 = ConvBlock(4* dim, 3 , 3, 1, 1)
        for m in self.modules():#自定义参数初始化 self.children()存储网络结构的子模块,只包括网络模块的第一代儿子模块，而self.modules()采用深度优先遍历包含网络模块的自己本身和所有的后代模块
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        for i in range(3):
            list_tensor.append(DMT1_yh[0][:, :, i, :, :])
        list_tensor.append(DMT1_yl)
        return torch.cat(list_tensor, 1)
    def _Itransformer(self, out):
        yh = []
        C = int(out.shape[1] / 4)
        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yl = y[:, :, 0].contiguous()
        yh.append(y[:, :, 1:].contiguous())
        return yl, yh
    def forward(self, x):
        out1 = self.feat1(x)#batch*32*192*192
        DMT1_yl,DMT1_yh=self.DWT(out1)
        out2=self._transformer(DMT1_yl,DMT1_yh)#batch*128*96*96
        out2=self.feat2(out2)#batch*64*96*96
        DMT2_yl, DMT2_yh = self.DWT(out2)
        out3 = self._transformer(DMT2_yl, DMT2_yh)#batch*256*48*48
        out3 = self.feat3(out3)#batch*128*48*48
        DMT3_yl, DMT3_yh = self.DWT(out3)
        out4 = self._transformer(DMT3_yl, DMT3_yh)#batch*512*24*24
        out4 = self.feat4(out4)#batch*256*24*24
        GSA=self.GSA(out4)#batch*256*24*24
        out5=self._Itransformer(GSA)
        out5=self.IDWT(out5) #batch*64*48*48
        out6=self.feat5(out5)#batch*128*48*48
        basic_1 = self.MGDB_1(out3)#batch*128*48*48
        out7 = torch.cat((out6,basic_1),1)#batch*256*48*48
        out8 = self.feat6(out7)#batch*128*48*48
        out9=self._Itransformer(out8)##batch*32*96*96
        out9=self.IDWT(out9)
        out10=self.feat7(out9)##batch*64*96*96
        basic_2 = self.MGDB_2(out2)##batch*64*96*96
        out11 = torch.cat((out10, basic_2),1)##batch*128*96*96
        out12 = self.feat8(out11)##batch*64*96*96
        out13 = self._Itransformer(out12)
        out14 = self.IDWT(out13)##batch*16*96*96
        out15 = self.feat9(out14)##batch*32*96*96
        out16=torch.cat((out1,out15),1)
        out17=self.MGDB_3(out16)
        out18=torch.cat((out16,out17),1)
        out19=self.MGDB_4(out18)
        out20=torch.cat((out18,out19),1)
        out21=self.feat10(out20)
        pred=self.feat11(out21)
        return pred

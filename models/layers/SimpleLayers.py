
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_weights

class UnetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, ks=3, n=2, stride=1, padding=1):
        super(UnetConv2, self).__init__()
        self.stride = stride
        self.padding = padding
        self.n = n

        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks,
                                               self.stride, self.padding),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True))
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size
        else:
            for i in range(1, n+1):
                conv= nn.Sequential(nn.Conv2d(in_size, out_size, ks,
                                              self.stride, self.padding),
                                    nn.ReLU(inplace=True))
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)
        return x


class UnetUp2(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(UnetUp2, self).__init__()
        self.conv = UnetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        for m in self.children():
            # if m.__class__.__name__.find('UnetConv2') != -1: continue;
            if m.__class__.__name__.find('ConvTranspose') != -1:
                init_weights(m,init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset//2 , offset//2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))
